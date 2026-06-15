import os
import time
import random
import psycopg2
from datetime import datetime, timezone

INTERVAL = 2 # Запись каждые 2 секунды

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname":   os.getenv("POSTGRES_DB", "metrics_db"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def generate_cpu(step):
    base = 40
    wave = 20 * (0.5 - random.random())
    trend = step * 0.01
    noise = random.uniform(-5, 5)
    spike = random.uniform(20, 45) if random.random() < 0.05 else 0
    value = base + wave + trend + noise + spike
    return max(5, min(98, round(value, 2)))

def generate_ram(cpu):
    ram = cpu * 0.75 + random.uniform(5, 15)
    return max(10, min(95, round(ram, 2)))

def generate_disk(step):
    value = 45 + step * 0.015 + random.uniform(-1, 1)
    return max(40, min(90, round(value, 2)))

def generate_network(cpu):
    rx = cpu * random.uniform(1000, 3000)
    tx = cpu * random.uniform(800, 2500)
    return round(rx, 2), round(tx, 2)

def insert_metrics(conn, row):
    sql = """
        INSERT INTO metrics (
            collected_at, cpu_usage, ram_usage, disk_usage, net_rx_bytes, net_tx_bytes
        ) VALUES (
            %(collected_at)s, %(cpu_usage)s, %(ram_usage)s, %(disk_usage)s, %(net_rx_bytes)s, %(net_tx_bytes)s
        )
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, row)
        conn.commit()
    except Exception as e:
        print(f"Ошибка БД: {e}")
        conn.rollback()

def main():
    print("Запуск генератора...")
    while True:
        try:
            conn = get_connection()
            print("Подключено к БД!")
            break
        except psycopg2.OperationalError:
            print("Ожидание БД...")
            time.sleep(2)

    step = 0
    while True:
        cpu = generate_cpu(step)
        ram = generate_ram(cpu)
        disk = generate_disk(step)
        rx, tx = generate_network(cpu)

        row = {
            "collected_at": datetime.now(timezone.utc),
            "cpu_usage": cpu,
            "ram_usage": ram,
            "disk_usage": disk,
            "net_rx_bytes": rx,
            "net_tx_bytes": tx,
        }

        insert_metrics(conn, row)
        print(f"Записано: CPU={cpu}% RAM={ram}%")

        step += 1
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
