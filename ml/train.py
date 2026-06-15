import os, pickle, logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import psycopg2, psycopg2.extras
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname": os.getenv("POSTGRES_DB", "metrics_db"),
}

LAGS = 60
TARGET = "cpu_usage"
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")

def load_data():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT collected_at, cpu_usage, ram_usage, disk_usage, net_rx_bytes, net_tx_bytes FROM metrics ORDER BY collected_at ASC")
            rows = cur.fetchall()
    finally: conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    return df.set_index("collected_at").ffill().fillna(0)

def build_features(df, lags):
    records = []
    values = df[TARGET].values
    for i in range(lags, len(values)):
        window = values[i-lags:i]
        feat = {f"lag_{j+1}": window[lags-j-1] for j in range(lags)}
        feat.update({
            "mean": np.mean(window), "std": np.std(window), "max": np.max(window), "min": np.min(window),
            "hour": df.index[i].hour, "weekday": df.index[i].weekday(),
            "ram": df["ram_usage"].iloc[i], "disk": df["disk_usage"].iloc[i],
            "rx": df["net_rx_bytes"].iloc[i], "tx": df["net_tx_bytes"].iloc[i]
        })
        records.append((list(feat.values()), values[i]))
    return np.array([r[0] for r in records]), np.array([r[1] for r in records])

def main():
    df = load_data()
    X, y = build_features(df, LAGS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = lgb.LGBMRegressor(n_estimators=300, random_state=42, verbose=-1)
    model.fit(X_scaled, y)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "lags": LAGS}, f)
    log.info("Модель обучена и сохранена. Признаков: %d", X.shape[1])

if name == "__main__":
    main()
