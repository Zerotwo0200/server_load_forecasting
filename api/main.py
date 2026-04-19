import os
import pickle
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Конфиг ───────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname":   os.getenv("POSTGRES_DB", "metrics_db"),
}
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Thesis Forecasting API",
    description="ML-система прогнозирования нагрузки серверной инфраструктуры",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Модели данных ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    metric: str        # cpu_usage / ram_usage / disk_usage
    steps: int = 6     # сколько шагов вперёд (каждый шаг = 1 минута)

class PredictResponse(BaseModel):
    metric: str
    steps: int
    predictions: list[float]
    timestamps: list[str]
    model_version: str

class MetricRow(BaseModel):
    id: int
    collected_at: str
    cpu_usage: float
    ram_usage: float
    disk_usage: float
    net_rx_bytes: Optional[float]
    net_tx_bytes: Optional[float]

class HealthResponse(BaseModel):
    status: str
    db: str
    model_loaded: bool
    timestamp: str

# ── БД ────────────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)

def check_db() -> bool:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False

def fetch_recent_metrics(limit: int = 100) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, collected_at, cpu_usage, ram_usage,
                       disk_usage, net_rx_bytes, net_tx_bytes
                FROM metrics
                ORDER BY collected_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def fetch_metric_series(metric_name: str, limit: int = 200) -> list[float]:
    allowed = {"cpu_usage", "ram_usage", "disk_usage"}
    if metric_name not in allowed:
        raise HTTPException(status_code=400, detail=f"Метрика должна быть одной из: {allowed}")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {metric_name} FROM metrics ORDER BY collected_at DESC LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
        # Возвращаем в хронологическом порядке (от старого к новому)
        return [float(r[metric_name]) for r in reversed(rows)]
    finally:
        conn.close()

def save_predictions(metric: str, preds: list[float], timestamps: list[str], version: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for ts, val in zip(timestamps, preds):
                cur.execute(
                    """
                    INSERT INTO predictions (predicted_at, target_time, metric_name, predicted_value, model_version)
                    VALUES (NOW(), %s, %s, %s, %s)
                    """,
                    (ts, metric, val, version)
                )
        conn.commit()
    finally:
        conn.close()

# ── Модель ────────────────────────────────────────────────────────────────────
_model_cache: dict = {}

def load_model() -> dict | None:
    if _model_cache:
        return _model_cache
    if not os.path.exists(MODEL_PATH):
        log.warning("Файл модели не найден: %s", MODEL_PATH)
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        _model_cache.update(data)
        log.info("Модель загружена. Версия: %s", data.get("version", "unknown"))
        return _model_cache
    except Exception as e:
        log.error("Ошибка загрузки модели: %s", e)
        return None

def make_features(series: list[float], lags: int = 6, ram: float = 0.0, disk: float = 0.0) -> np.ndarray:
    """Строим признаки из последних значений ряда — должны совпадать с train.py."""
    arr = np.array(series[-lags:], dtype=float)
    now = datetime.now(timezone.utc)
    features = list(arr)
    features.append(float(np.mean(arr)))
    features.append(float(np.std(arr)))
    features.append(float(np.max(arr)))
    features.append(float(np.min(arr)))
    features.append(float(now.hour))
    features.append(float(now.weekday()))
    features.append(float(ram))
    features.append(float(disk))
    return np.array(features).reshape(1, -1)

def predict_steps(series: list[float], model, scaler, steps: int, lags: int = 6,
                  ram_series: list[float] = None, disk_series: list[float] = None) -> list[float]:
    """Итеративный прогноз: каждый новый прогноз добавляется в ряд."""
    extended = list(series)
    ram  = float(ram_series[-1])  if ram_series  else 0.0
    disk = float(disk_series[-1]) if disk_series else 0.0
    results = []
    for _ in range(steps):
        features = make_features(extended, lags, ram=ram, disk=disk)
        features_scaled = scaler.transform(features)
        pred = float(model.predict(features_scaled)[0])
        pred = round(max(0.0, min(100.0, pred)), 2)
        results.append(pred)
        extended.append(pred)
    return results

# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    model_data = load_model()
    return HealthResponse(
        status="ok",
        db="ok" if check_db() else "unavailable",
        model_loaded=model_data is not None,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/metrics", response_model=list[MetricRow], tags=["Metrics"])
def get_metrics(limit: int = Query(default=100, ge=1, le=1000)):
    """Последние N записей метрик из БД."""
    rows = fetch_recent_metrics(limit)
    if not rows:
        raise HTTPException(status_code=404, detail="Данных пока нет. Подожди пока collector соберёт метрики.")
    return [
        MetricRow(
            id=r["id"],
            collected_at=r["collected_at"].isoformat(),
            cpu_usage=r["cpu_usage"],
            ram_usage=r["ram_usage"],
            disk_usage=r["disk_usage"],
            net_rx_bytes=r.get("net_rx_bytes"),
            net_tx_bytes=r.get("net_tx_bytes"),
        )
        for r in rows
    ]

@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(req: PredictRequest):
    """Прогноз метрики на steps шагов вперёд (шаг = 1 минута)."""
    if req.steps < 1 or req.steps > 60:
        raise HTTPException(status_code=400, detail="steps должен быть от 1 до 60.")

    model_data = load_model()
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Модель ещё не обучена. Сначала запусти ml/train.py после накопления данных."
        )

    series = fetch_metric_series(req.metric, limit=200)
    if len(series) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Недостаточно данных для прогноза. Нужно минимум 10 записей, сейчас: {len(series)}."
        )

    model  = model_data["model"]
    scaler = model_data["scaler"]
    version = model_data.get("version", "unknown")
    lags   = model_data.get("lags", 6)

    ram_series  = fetch_metric_series("ram_usage",  limit=10) if req.metric != "ram_usage"  else series
    disk_series = fetch_metric_series("disk_usage", limit=10) if req.metric != "disk_usage" else series
    preds = predict_steps(series, model, scaler, req.steps, lags,
                          ram_series=ram_series, disk_series=disk_series)

    now = datetime.now(timezone.utc)
    timestamps = [
        (now.replace(second=0, microsecond=0).timestamp() + i * 60)
        for i in range(1, req.steps + 1)
    ]
    ts_iso = [datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in timestamps]

    save_predictions(req.metric, preds, ts_iso, version)

    return PredictResponse(
        metric=req.metric,
        steps=req.steps,
        predictions=preds,
        timestamps=ts_iso,
        model_version=version
    )

@app.get("/predictions", tags=["Forecasting"])
def get_predictions(metric: str = "cpu_usage", limit: int = Query(default=50, ge=1, le=500)):
    """История прогнозов из БД (для Grafana)."""
    allowed = {"cpu_usage", "ram_usage", "disk_usage"}
    if metric not in allowed:
        raise HTTPException(status_code=400, detail=f"Метрика должна быть одной из: {allowed}")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT predicted_at, target_time, metric_name, predicted_value, model_version
                FROM predictions
                WHERE metric_name = %s
                ORDER BY target_time DESC
                LIMIT %s
                """,
                (metric, limit)
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Рекомендации ──────────────────────────────────────────────────────────────

THRESHOLDS = {
    "scale_up":   75.0,   # % — выше этого включаем доп. серверы
    "scale_down": 30.0,   # % — ниже этого отключаем лишние серверы
    "critical":   90.0,   # % — критический уровень
}

class ServerAction(BaseModel):
    action: str           # scale_up / scale_down / hold
    reason: str
    servers_delta: int    # +N включить / -N отключить / 0 ничего
    urgency: str          # low / medium / high / critical
    predicted_peak: float
    predicted_min: float

@app.post("/recommendations", response_model=ServerAction, tags=["Recommendations"])
def get_recommendations(req: PredictRequest):
    """
    На основе прогноза нагрузки выдаёт рекомендацию:
    нужно ли включать или отключать серверы.
    """
    # Получаем прогноз
    model_data = load_model()
    if model_data is None:
        raise HTTPException(status_code=503, detail="Модель не загружена.")

    series = fetch_metric_series(req.metric, limit=200)
    if len(series) < 10:
        raise HTTPException(status_code=422, detail="Недостаточно данных.")

    model   = model_data["model"]
    scaler  = model_data["scaler"]
    lags    = model_data.get("lags", 6)

    ram_series  = fetch_metric_series("ram_usage",  limit=10)
    disk_series = fetch_metric_series("disk_usage", limit=10)
    preds = predict_steps(series, model, scaler, req.steps, lags,
                          ram_series=ram_series, disk_series=disk_series)

    peak = max(preds)
    low  = min(preds)

    # Логика рекомендаций
    if peak >= THRESHOLDS["critical"]:
        action        = "scale_up"
        servers_delta = 3
        urgency       = "critical"
        reason        = f"Прогнозируется критическая нагрузка {peak:.1f}% — срочно нужны дополнительные серверы."
    elif peak >= THRESHOLDS["scale_up"]:
        action        = "scale_up"
        servers_delta = 1
        urgency       = "medium" if peak < 85 else "high"
        reason        = f"Прогнозируемая нагрузка {peak:.1f}% превышает порог {THRESHOLDS['scale_up']}%. Рекомендуется включить {servers_delta} сервер(а)."
    elif low <= THRESHOLDS["scale_down"]:
        action        = "scale_down"
        servers_delta = -1
        urgency       = "low"
        reason        = f"Прогнозируемая нагрузка упадёт до {low:.1f}%. Можно отключить {abs(servers_delta)} сервер для экономии энергии."
    else:
        action        = "hold"
        servers_delta = 0
        urgency       = "low"
        reason        = f"Нагрузка в норме: прогноз {low:.1f}%–{peak:.1f}%. Изменений не требуется."

    # Сохраняем в БД для истории
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recommendations (metric_name, action, servers_delta, urgency, predicted_peak, predicted_min, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (req.metric, action, servers_delta, urgency, peak, low, reason)
            )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()

    return ServerAction(
        action=action,
        reason=reason,
        servers_delta=servers_delta,
        urgency=urgency,
        predicted_peak=round(peak, 2),
        predicted_min=round(low, 2),
    )
