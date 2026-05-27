import os, pickle, logging, threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import psycopg2, psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB = dict(host=os.getenv("POSTGRES_HOST","postgres"),
          port=int(os.getenv("POSTGRES_PORT","5432")),
          user=os.getenv("POSTGRES_USER","thesis"),
          password=os.getenv("POSTGRES_PASSWORD","thesis_pass"),
          dbname=os.getenv("POSTGRES_DB","metrics_db"))
MODEL_PATH = os.getenv("MODEL_PATH","/app/model/model.pkl")

# Горизонты автопрогнозирования (шагов → минут при интервале 1 мин)
FORECAST_STEPS = [5, 10, 15, 30, 60]
METRICS = ["cpu_usage", "ram_usage", "disk_usage"]

app = FastAPI(title="Thesis Forecasting API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Модели Pydantic ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    metric: str
    steps: int = 15

class AlertConfigCreate(BaseModel):
    metric_name: str
    steps: int
    threshold: float
    condition: str = "above"   # above | below

class AlertConfigOut(BaseModel):
    id: int
    metric_name: str
    steps: int
    threshold: float
    condition: str
    enabled: bool
    created_at: str

# ── БД ────────────────────────────────────────────────────────────────
def conn():
    return psycopg2.connect(**DB,
                            cursor_factory=psycopg2.extras.RealDictCursor)

def db_ok():
    try:
        c = conn(); c.cursor().execute("SELECT 1"); c.close(); return True
    except: return False

def fetch_series(metric: str, limit: int = 200):
    allowed = {"cpu_usage","ram_usage","disk_usage"}
    if metric not in allowed:
        raise HTTPException(400, f"metric must be one of {allowed}")
    c = conn()
    try:
        cur = c.cursor()
        cur.execute(f"SELECT {metric} FROM metrics "
                    f"ORDER BY collected_at DESC LIMIT %s", (limit,))
        return [float(r[metric]) for r in reversed(cur.fetchall())]
    finally:
        c.close()

def save_predictions(metric, preds, timestamps, version):
    c = conn()
    try:
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO predictions "
                "(predicted_at,target_time,metric_name,predicted_value,model_version)"
                " VALUES (NOW(),%s,%s,%s,%s)",
                [(ts, metric, val, version) for ts, val in zip(timestamps, preds)]
            )
        c.commit()
    finally:
        c.close()

# ── Модель ────────────────────────────────────────────────────────────
_cache: dict = {}
_lock = threading.Lock()

def load_model():
    with _lock:
        if _cache: return _cache
        if not os.path.exists(MODEL_PATH):
            return None
        try:
            with open(MODEL_PATH,"rb") as f:
                data = pickle.load(f)
            _cache.update(data)
            log.info("Модель загружена: %s", data.get("version"))
            return _cache
        except Exception as e:
            log.error("Ошибка загрузки модели: %s", e)
            return None

def reload_model():
    """Перезагрузить модель (если файл обновился)"""
    with _lock:
        _cache.clear()
    return load_model()

def make_features(series, lags=6, ram=0.0, disk=0.0):
    arr = np.array(series[-lags:], dtype=float)
    now = datetime.now(timezone.utc)
    return np.array(
        list(arr) +
        [float(np.mean(arr)), float(np.std(arr)),
         float(np.max(arr)), float(np.min(arr)),
         float(now.hour), float(now.weekday()),
         ram, disk]
    ).reshape(1, -1)

def predict_steps(series, model, scaler, steps, lags=6,
                  ram_series=None, disk_series=None):
    extended = list(series)
    ram  = float(ram_series[-1])  if ram_series  else 0.0
    disk = float(disk_series[-1]) if disk_series else 0.0
    results = []
    for _ in range(steps):
        feat = make_features(extended, lags, ram, disk)
        pred = float(model.predict(scaler.transform(feat))[0])
        pred = round(max(0.0, min(100.0, pred)), 2)
        results.append(pred)
        extended.append(pred)
    return results

def run_forecast(metric: str, steps: int):
    """Запустить прогноз и сохранить в БД"""
    md = load_model()
    if md is None: return
    try:
        series = fetch_series(metric, 200)
        if len(series) < 10: return
        ram  = fetch_series("ram_usage",  10) if metric != "ram_usage"  else series
        disk = fetch_series("disk_usage", 10) if metric != "disk_usage" else series
        preds = predict_steps(series, md["model"], md["scaler"],
                              steps, md.get("lags",6), ram, disk)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        timestamps = [now + timedelta(minutes=i+1) for i in range(steps)]
        save_predictions(metric, preds, timestamps, md.get("version","?"))
    except Exception as e:
        log.error("Ошибка прогноза %s/%d: %s", metric, steps, e)

# ── Планировщик ───────────────────────────────────────────────────────
def scheduled_forecast():
    """Каждую минуту строим прогнозы для всех метрик и горизонтов"""
    for metric in METRICS:
        for steps in FORECAST_STEPS:
            run_forecast(metric, steps)
    check_alerts()

def check_alerts():
    """Проверить все активные алерты"""
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM alert_configs WHERE enabled = TRUE")
            configs = cur.fetchall()
        for cfg in configs:
            metric  = cfg["metric_name"]
            steps   = cfg["steps"]
            thresh  = cfg["threshold"]
            cond    = cfg["condition"]
            now     = datetime.now(timezone.utc)
            horizon = now + timedelta(minutes=steps)
            with c.cursor() as cur:
                cur.execute(
                    """SELECT AVG(predicted_value) AS avg_val
                       FROM predictions
                       WHERE metric_name = %s
                         AND target_time BETWEEN %s AND %s
                         AND predicted_at >= NOW() - INTERVAL '2 minutes'""",
                    (metric, now, horizon)
                )
                row = cur.fetchone()
            if not row or row["avg_val"] is None:
                continue
            val = float(row["avg_val"])
            triggered = (cond == "above" and val > thresh) or \
                        (cond == "below" and val < thresh)
            if triggered:
                msg = (f"Прогноз {metric} через {steps} мин: "
                       f"{val:.1f}% {'>' if cond=='above' else '<'} "
                       f"порог {thresh}%")
                with c.cursor() as cur:
                    cur.execute(
                        """INSERT INTO alert_events
                           (metric_name,steps,threshold,predicted_value,message,config_id)
                           VALUES (%s,%s,%s,%s,%s,%s)""",
                        (metric, steps, thresh, val, msg, cfg["id"])
                    )
                c.commit()
                log.warning("ALERT: %s", msg)
    except Exception as e:
        log.error("Ошибка проверки алертов: %s", e)
    finally:
        c.close()

# ── Запуск планировщика ───────────────────────────────────────────────
@app.on_event("startup")
def startup():
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_forecast, "interval", minutes=1,
                      id="forecast", next_run_time=datetime.now())
    scheduler.start()
    log.info("Планировщик запущен")

# ── Эндпоинты ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status":"ok", "db": "ok" if db_ok() else "error",
            "model_loaded": load_model() is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/metrics")
def get_metrics(limit: int = Query(100, ge=1, le=1000)):
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT id,collected_at,cpu_usage,ram_usage,"
                "disk_usage,net_rx_bytes,net_tx_bytes "
                "FROM metrics ORDER BY collected_at DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()

@app.post("/predict")
def predict(req: PredictRequest):
    if req.steps < 1 or req.steps > 120:
        raise HTTPException(400, "steps: 1–120")
    md = load_model()
    if md is None:
        raise HTTPException(503, "Модель не загружена")
    series = fetch_series(req.metric, 200)
    if len(series) < 10:
        raise HTTPException(422, "Мало данных")
    ram  = fetch_series("ram_usage",  10) if req.metric != "ram_usage"  else series
    disk = fetch_series("disk_usage", 10) if req.metric != "disk_usage" else series
    preds = predict_steps(series, md["model"], md["scaler"],
                          req.steps, md.get("lags",6), ram, disk)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    ts  = [(now + timedelta(minutes=i+1)).isoformat() for i in range(req.steps)]
    save_predictions(req.metric, preds, ts, md.get("version","?"))
    return {"metric": req.metric, "steps": req.steps,
            "predictions": preds, "timestamps": ts,
            "model_version": md.get("version")}

@app.get("/predictions")
def get_predictions(metric: str = "cpu_usage",
                    steps: int = Query(15, ge=1, le=120)):
    """Последний прогноз для заданного горизонта"""
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                """SELECT target_time, predicted_value, model_version
                   FROM predictions
                   WHERE metric_name = %s
                     AND target_time >= NOW()
                     AND target_time <= NOW() + make_interval(mins => %s)
                     AND predicted_at >= NOW() - INTERVAL '2 minutes'
                   ORDER BY target_time""",
                (metric, steps))
            return [dict(r) for r in cur.fetchall()]
    finally:
        c.close()

# ── Alert config endpoints ────────────────────────────────────────────
@app.get("/alerts/configs")
def list_alert_configs():
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM alert_configs ORDER BY id")
            return [dict(r) for r in cur.fetchall()]
    finally:
        c.close()

@app.post("/alerts/configs", status_code=201)
def create_alert_config(body: AlertConfigCreate):
    if body.condition not in ("above","below"):
        raise HTTPException(400, "condition must be 'above' or 'below'")
    if body.steps not in FORECAST_STEPS:
        raise HTTPException(400, f"steps must be one of {FORECAST_STEPS}")
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO alert_configs "
                "(metric_name,steps,threshold,condition) "
                "VALUES (%s,%s,%s,%s) RETURNING *",
                (body.metric_name, body.steps, body.threshold, body.condition))
            row = dict(cur.fetchone())
        c.commit()
        return row
    finally:
        c.close()

@app.delete("/alerts/configs/{config_id}", status_code=204)
def delete_alert_config(config_id: int):
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute("DELETE FROM alert_configs WHERE id = %s", (config_id,))
        c.commit()
    finally:
        c.close()

@app.get("/alerts/events")
def get_alert_events(limit: int = Query(50, ge=1, le=500)):
    c = conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT * FROM alert_events "
                "ORDER BY triggered_at DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        c.close()
