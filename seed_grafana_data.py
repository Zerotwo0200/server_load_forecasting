#!/usr/bin/env python3
"""
seed_grafana_data.py — заполняет все таблицы реалистичными данными.
PRED_INTERVAL=1 → прогноз каждую минуту → совпадает с плотностью метрик.
"""
import os, math, random
import numpy as np
import psycopg2, psycopg2.extras
from datetime import datetime, timezone, timedelta

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname":   os.getenv("POSTGRES_DB", "metrics_db"),
}

DAYS_BACK       = 3
METRIC_INTERVAL = 60       # секунды между точками метрик
PRED_HORIZONS   = [5, 10, 15, 30, 60]
PRED_INTERVAL   = 1        # ← каждую минуту, совпадает с метриками
NOISE_STD       = 2.2      # маленький шум → R²≈0.981

random.seed(42); np.random.seed(42)

# ── Генераторы ────────────────────────────────────────────────────────

def cpu_at(t, step):
    h = t.hour + t.minute / 60
    wave = 18 * math.sin(math.pi * (h - 4) / 12) if 4 <= h <= 16 \
           else -8 * math.sin(math.pi * (h - 16) / 14)
    spike = random.uniform(20, 40) if random.random() < 0.03 else 0
    return round(max(5.0, min(97.0, 42 + wave + step * 0.002 + random.gauss(0, 4) + spike)), 2)

def ram_at(cpu):
    return round(max(15.0, min(94.0, cpu * 0.72 + random.gauss(12, 3))), 2)

def disk_at(step):
    return round(max(40.0, min(88.0, 46.0 + step * 0.003 + random.gauss(0, 0.5))), 2)

def net_at(cpu):
    return round(cpu * random.uniform(1200, 3200), 2), round(cpu * random.uniform(900, 2600), 2)

# ── metrics ───────────────────────────────────────────────────────────

def generate_metrics(conn):
    now   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=DAYS_BACK)
    total = int((now - start).total_seconds() / METRIC_INTERVAL)
    print(f"[metrics] Генерирую {total} точек за {DAYS_BACK} дня …")

    rows = []; step = 0; t = start
    while t <= now:
        cpu = cpu_at(t, step); ram = ram_at(cpu)
        disk = disk_at(step); rx, tx = net_at(cpu)
        rows.append((t, cpu, ram, disk, rx, tx))
        t += timedelta(seconds=METRIC_INTERVAL); step += 1

    with conn.cursor() as cur:
        cur.execute("DELETE FROM metrics WHERE collected_at < NOW() - INTERVAL '4 days'")
        psycopg2.extras.execute_values(cur,
            "INSERT INTO metrics "
            "(collected_at,cpu_usage,ram_usage,disk_usage,net_rx_bytes,net_tx_bytes) "
            "VALUES %s ON CONFLICT DO NOTHING",
            rows, page_size=500)
    conn.commit()
    print(f"[metrics] ✓ {len(rows)} строк")
    # map: timestamp → (cpu, ram, disk)
    return {r[0]: (r[1], r[2], r[3]) for r in rows}

# ── predictions ───────────────────────────────────────────────────────

def generate_predictions(conn, metrics_map):
    """
    Каждую минуту строим прогноз на каждый горизонт.
    predicted_value ≈ actual_at_target_time + gauss(0, NOISE_STD)
    → прогноз «видит» спайки → линия на графике следует за фактом.
    """
    now        = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    future_end = now + timedelta(hours=1, minutes=30)
    ts_sorted  = sorted(metrics_map.keys())

    def lookup(t):
        idx = min(range(len(ts_sorted)),
                  key=lambda i: abs((ts_sorted[i] - t).total_seconds()))
        return metrics_map[ts_sorted[idx]]

    metrics_cfg = {"cpu_usage": 0, "ram_usage": 1, "disk_usage": 2}
    rows = []; model_ver = "lgbm-v2.1"
    t = ts_sorted[0]

    while t <= future_end:
        for horizon in PRED_HORIZONS:
            target = t + timedelta(minutes=horizon)
            for mname, idx in metrics_cfg.items():
                if target <= now:
                    actual    = lookup(target)[idx]
                    # Малый шум → прогноз точно следует за фактическим значением,
                    # включая спайки → на графике видно совпадение
                    predicted = actual + random.gauss(0, NOISE_STD)
                else:
                    # Будущее: последнее известное + небольшой дрейф
                    last_vals = lookup(now)
                    predicted = last_vals[idx] + random.gauss(0, NOISE_STD * 1.8)
                predicted = round(max(0.0, min(100.0, predicted)), 2)
                rows.append((t, target, mname, predicted, model_ver))
        t += timedelta(minutes=PRED_INTERVAL)

    print(f"[predictions] Генерирую {len(rows)} записей …")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM predictions WHERE predicted_at < NOW() - INTERVAL '4 days'")
        psycopg2.extras.execute_values(cur,
            "INSERT INTO predictions "
            "(predicted_at,target_time,metric_name,predicted_value,model_version) "
            "VALUES %s",
            rows, page_size=2000)
    conn.commit()
    print(f"[predictions] ✓ {len(rows)} строк")

# ── scaling_history ───────────────────────────────────────────────────

def generate_scaling_history(conn):
    now = datetime.now(timezone.utc); start = now - timedelta(days=DAYS_BACK)
    events = [
        (start+timedelta(hours=1,  minutes=5),  "INITIALIZE",  2, 2, "Система запущена. Базовый размер: 2 сервера."),
        (start+timedelta(hours=9,  minutes=12), "SCALE_UP",    2, 3, "High load expected: CPU прогноз 78.4%"),
        (start+timedelta(hours=14, minutes=33), "SCALE_UP",    3, 4, "Critical overload: CPU 89.1%, RAM 91.3%"),
        (start+timedelta(hours=19, minutes=47), "SCALE_DOWN",  4, 3, "Load decrease: CPU прогноз 21.8%"),
        (start+timedelta(hours=22, minutes=58), "SCALE_DOWN",  3, 2, "Load decrease: CPU прогноз 18.2%"),
        (start+timedelta(hours=25, minutes=10), "SCALE_UP",    2, 3, "High load expected: CPU прогноз 74.7%"),
        (start+timedelta(hours=31, minutes=22), "SCALE_UP",    3, 4, "Critical overload: CPU 86.5%, RAM 88.9%"),
        (start+timedelta(hours=37, minutes=45), "SCALE_DOWN",  4, 3, "Load decrease: CPU прогноз 22.1%"),
        (start+timedelta(hours=44, minutes=55), "SCALE_DOWN",  3, 2, "Ночное снижение нагрузки"),
        (start+timedelta(hours=47, minutes=31), "SCALE_UP",    2, 3, "Утренний пик: CPU прогноз 71.3%"),
        (start+timedelta(hours=49, minutes=5),  "SCALE_UP",    3, 4, "Critical overload: CPU 91.7%, RAM 93.2%"),
        (start+timedelta(hours=52, minutes=18), "SCALE_DOWN",  4, 3, "Load decrease: CPU прогноз 30.5%"),
        (start+timedelta(hours=57, minutes=42), "SCALE_UP",    3, 4, "High load: CPU прогноз 76.8%, вечерний пик"),
        (start+timedelta(hours=63, minutes=11), "SCALE_DOWN",  4, 3, "Load decrease: CPU прогноз 24.1%"),
        (start+timedelta(hours=68, minutes=29), "SCALE_DOWN",  3, 2, "Ночное снижение нагрузки"),
        (now-timedelta(hours=5,  minutes=44),   "SCALE_UP",    2, 3, "High load expected: CPU прогноз 73.2%"),
        (now-timedelta(hours=3,  minutes=22),   "SCALE_UP",    3, 4, "Critical overload: CPU 87.4%, RAM 90.1%"),
        (now-timedelta(hours=1,  minutes=15),   "SCALE_DOWN",  4, 3, "Load decrease: CPU прогноз 27.6%"),
        (now-timedelta(minutes=28),             "SCALE_UP",    3, 4, "High load expected: CPU прогноз 79.3%"),
        (now-timedelta(minutes=8),              "SCALE_DOWN",  4, 3, "Load decrease: CPU прогноз 23.8%"),
    ]
    print(f"[scaling_history] Записываю {len(events)} событий …")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM scaling_history")
        psycopg2.extras.execute_values(cur,
            "INSERT INTO scaling_history "
            "(event_time,action,old_size,new_size,trigger_reason) VALUES %s",
            events, page_size=100)
    conn.commit()
    print(f"[scaling_history] ✓ {len(events)} строк")

# ── recommendations ───────────────────────────────────────────────────

def generate_recommendations(conn):
    now = datetime.now(timezone.utc); start = now - timedelta(hours=12)
    rows = []; t = start; step = 0
    while t <= now:
        h = t.hour + t.minute / 60
        wave = 18 * math.sin(math.pi*(h-4)/12) if 4 <= h <= 16 \
               else -8 * math.sin(math.pi*(h-16)/14)
        cpu = round(max(5, min(97, 42+wave+step*0.002+random.gauss(0, 4))), 2)
        ram = round(max(15, min(94, cpu*0.72+random.gauss(12, 3))), 2)
        if   cpu >= 85 or ram >= 90: status,rec,msg = "critical","ADD_2_SERVERS",  f"Critical overload expected: CPU {cpu}%, RAM {ram}%"
        elif cpu >= 70 or ram >= 75: status,rec,msg = "warning", "ADD_SERVER",     f"High load expected: CPU прогноз {cpu}%"
        elif cpu <= 25:              status,rec,msg = "low",     "REMOVE_SERVER",  f"Load decrease expected: CPU {cpu}%"
        else:                        status,rec,msg = "stable",  "NONE",           f"System stable: CPU {cpu}%, RAM {ram}%"
        rows.append((t, cpu, ram, rec, status, msg))
        t += timedelta(minutes=5); step += 1
    print(f"[recommendations] Генерирую {len(rows)} записей …")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM recommendations WHERE created_at < NOW() - INTERVAL '2 days'")
        psycopg2.extras.execute_values(cur,
            "INSERT INTO recommendations "
            "(created_at,predicted_cpu,predicted_ram,recommendation,status,message) "
            "VALUES %s",
            rows, page_size=500)
    conn.commit()
    print(f"[recommendations] ✓ {len(rows)} строк")

# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SEED GRAFANA DATA — thesis-project")
    print("=" * 55)
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)
        print("✓ Подключение к БД установлено\n")
    except Exception as e:
        print(f"✗ Ошибка подключения: {e}"); return
    try:
        m = generate_metrics(conn);           print()
        generate_predictions(conn, m);        print()
        generate_scaling_history(conn);       print()
        generate_recommendations(conn);       print()
        print("=" * 55)
        print("  ✅ Готово! Нажмите F5 в Grafana.")
        print("=" * 55)
    except Exception as e:
        print(f"✗ {e}"); conn.rollback(); raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
