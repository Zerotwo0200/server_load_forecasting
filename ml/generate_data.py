import os
import math
import random
import psycopg2
from datetime import datetime, timezone, timedelta

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname":   os.getenv("POSTGRES_DB", "metrics_db"),
}

def cpu_load(dt: datetime) -> float:
    hour    = dt.hour
    weekday = dt.weekday()  # 0=пн, 4=пт, 5-6=выходные

    # Базовая нагрузка по времени суток
    if 0 <= hour < 6:
        base = 5.0
    elif 6 <= hour < 9:
        base = 25.0 + (hour - 6) * 10   # утренний рост
    elif 9 <= hour < 12:
        base = 55.0
    elif 12 <= hour < 14:
        base = 45.0                      # обед — чуть легче
    elif 14 <= hour < 18:
        base = 65.0                      # послеобеденный пик
    elif 18 <= hour < 20:
        base = 40.0
    else:
        base = 15.0

    # Выходные — нагрузка ниже
    if weekday >= 5:
        base *= 0.35

    # Пятница — пик в конце рабочего дня
    if weekday == 4 and 16 <= hour < 18:
        base += 20.0

    # Случайные всплески (раз в ~20 точек)
    spike = 30.0 if random.random() < 0.05 else 0.0

    noise = random.gauss(0, 3.0)
    return round(max(1.0, min(99.0, base + spike + noise)), 2)

def ram_load(cpu: float) -> float:
    # RAM коррелирует с CPU но с меньшей амплитудой
    base  = 30.0 + cpu * 0.4
    noise = random.gauss(0, 2.0)
    return round(max(10.0, min(95.0, base + noise)), 2)

def disk_load() -> float:
    # Диск растёт медленно
    return round(random.gauss(62.0, 1.5), 2)

def generate(days: int = 14, interval_minutes: int = 1):
    conn = psycopg2.connect(**DB_CONFIG)
    now  = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    rows = []
    t = start
    while t <= now:
        cpu  = cpu_load(t)
        ram  = ram_load(cpu)
        disk = disk_load()
        rows.append((t, cpu, ram, disk,
                     random.uniform(1e5, 5e6),
                     random.uniform(5e4, 2e6)))
        t += timedelta(minutes=interval_minutes)

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO metrics (collected_at, cpu_usage, ram_usage, disk_usage, net_rx_bytes, net_tx_bytes)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            rows
        )
    conn.commit()
    conn.close()
    print(f"Вставлено {len(rows)} строк ({days} дней с интервалом {interval_minutes} мин)")

if __name__ == "__main__":
    generate(days=14, interval_minutes=1)
