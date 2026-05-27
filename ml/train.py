import os
import pickle
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Конфиг ───────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname":   os.getenv("POSTGRES_DB", "metrics_db"),
}

LAGS        = 60          # количество лагов (признаки)
HORIZON     = 15          # шагов вперёд для обучения
TARGET      = "cpu_usage"  # основная целевая метрика
MODEL_PATH  = os.getenv("MODEL_PATH", "model/model.pkl")
PLOTS_DIR   = "plots"

# ── Загрузка данных ───────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT collected_at, cpu_usage, ram_usage, disk_usage,
                       net_rx_bytes, net_tx_bytes
                FROM metrics
                ORDER BY collected_at ASC
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("Таблица metrics пуста. Дай collector поработать минимум 20-30 минут.")

    df = pd.DataFrame([dict(r) for r in rows])
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    df = df.set_index("collected_at").sort_index()
    df = df.ffill().fillna(0)
    log.info("Загружено строк: %d  (с %s по %s)", len(df), df.index[0], df.index[-1])
    return df

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, target: str, lags: int, horizon: int) -> tuple:
    """
    Для каждой точки t строим вектор признаков:
      - лаги t-1 .. t-lags по целевой метрике
      - скользящее среднее, std, max, min по окну lags
      - час суток и день недели (временные признаки)
      - значения RAM и disk на момент t (корреляты)
    Целевая переменная: значение target через horizon шагов
    """
    records = []
    values  = df[target].values
    n       = len(values)

    for i in range(lags, n - horizon):
        window = values[i - lags : i]
        feat = {
            **{f"lag_{j+1}": window[lags - j - 1] for j in range(lags)},
            "roll_mean":  float(np.mean(window)),
            "roll_std":   float(np.std(window)),
            "roll_max":   float(np.max(window)),
            "roll_min":   float(np.min(window)),
            "hour":       df.index[i].hour,
            "weekday":    df.index[i].weekday(),
            "ram_usage":  float(df["ram_usage"].iloc[i]),
            "disk_usage": float(df["disk_usage"].iloc[i]),
            "net_rx": float(df["net_rx_bytes"].iloc[i]),
            "net_tx": float(df["net_tx_bytes"].iloc[i]),
        }
        label = float(values[i + horizon - 1])
        records.append((feat, label))

    X = pd.DataFrame([r[0] for r in records])
    y = np.array([r[1] for r in records])
    log.info("Построено примеров: %d  признаков: %d", len(X), X.shape[1])
    return X, y

# ── Обучение ──────────────────────────────────────────────────────────────────
def train(X: pd.DataFrame, y: np.ndarray) -> tuple:
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit — честная валидация, без утечки будущего в прошлое
    tscv    = TimeSeriesSplit(n_splits=5)
    metrics_cv = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx],        y[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        )

        preds = model.predict(X_val)
        residuals = y_val - preds
        std = np.std(residuals)
        mae  = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        r2   = r2_score(y_val, preds)
        metrics_cv.append({"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2})
        log.info("Fold %d — MAE: %.3f  RMSE: %.3f  R²: %.3f", fold, mae, rmse, r2)

    # Финальная модель — обучаем на всех данных
    final_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X_scaled, y)

    avg = {
        "MAE":  np.mean([m["MAE"]  for m in metrics_cv]),
        "RMSE": np.mean([m["RMSE"] for m in metrics_cv]),
        "R2":   np.mean([m["R2"]   for m in metrics_cv]),
    }
    log.info("Среднее по фолдам — MAE: %.3f  RMSE: %.3f  R²: %.3f", avg["MAE"], avg["RMSE"], avg["R2"])
    return final_model, scaler, avg

# ── Графики ───────────────────────────────────────────────────────────────────
def save_plots(model, X: pd.DataFrame, y: np.ndarray, scaler, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    X_scaled = scaler.transform(X)
    preds    = model.predict(X_scaled)

    # 1. Факт vs Прогноз
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y[-200:],     label="Факт",    linewidth=1.2, alpha=0.9)
    ax.plot(preds[-200:], label="Прогноз", linewidth=1.2, alpha=0.8, linestyle="--")
    ax.set_title(f"Факт vs Прогноз — {TARGET}")
    ax.set_ylabel("Нагрузка, %")
    ax.set_xlabel("Шаги")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "forecast_vs_actual.png"), dpi=120)
    plt.close(fig)

    # 2. Важность признаков
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig, ax  = plt.subplots(figsize=(8, 5))
    feat_imp.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Важность признаков (LightGBM)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=120)
    plt.close(fig)

    log.info("Графики сохранены в: %s/", plots_dir)

# ── Сохранение модели ─────────────────────────────────────────────────────────
def save_model(model, scaler, metrics: dict, lags: int):
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M")
    payload = {
        "model":   model,
        "scaler":  scaler,
        "lags":    lags,
        "target":  TARGET,
        "metrics": metrics,
        "version": version,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    log.info("Модель сохранена: %s  (версия %s)", MODEL_PATH, version)
    return version

# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    log.info("=== Обучение модели ===")

    df = load_data()

    if len(df) < 20:
        raise RuntimeError(
            f"Слишком мало данных: {len(df)} строк. "
            "Нужно минимум 20, лучше 100+. Подожди пока collector наберёт данные."
        )

    X, y = build_features(df, target=TARGET, lags=LAGS, horizon=HORIZON)
    model, scaler, avg_metrics = train(X, y)
    save_plots(model, X, y, scaler, PLOTS_DIR)
    version = save_model(model, scaler, avg_metrics, LAGS)

    log.info("=== Готово ===")
    log.info("Версия:  %s", version)
    log.info("MAE:     %.3f%%", avg_metrics["MAE"])
    log.info("RMSE:    %.3f%%", avg_metrics["RMSE"])
    log.info("R²:      %.4f",   avg_metrics["R2"])

if __name__ == "__main__":
    main()
