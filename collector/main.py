import os
import time
import logging
import requests
import psycopg2
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Конфиг из переменных окружения ───────────────────────────────────────────
PROMETHEUS_URL   = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
COLLECT_INTERVAL = int(os.getenv("COLLECT_INTERVAL", "60"))

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "thesis"),
    "password": os.getenv("POSTGRES_PASSWORD", "thesis_pass"),
    "dbname":   os.getenv("POSTGRES_DB", "metrics_db"),
}

# ── PromQL-запросы ────────────────────────────────────────────────────────────
QUERIES = {
    "cpu_usage": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)',
    "ram_usage": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
    "disk_usage": '(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}) / node_filesystem_size_bytes{mountpoint="/"} * 100',
    "net_rx_bytes": 'rate(node_network_receive_bytes_total{device!="lo"}[1m])',
    "net_tx_bytes": 'rate(node_network_transmit_bytes_total{device!="lo"}[1m])',
}

# ── Prometheus ────────────────────────────────────────────────────────────────
def query_prometheus(metric_name: str, promql: str) -> float | None:
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            log.warning("Пустой результат для метрики: %s", metric_name)
            return None
        # Суммируем значения если несколько результатов (напр. несколько сетевых интерфейсов)
        total = sum(float(r["value"][1]) for r in results)
        return round(total, 4)
    except Exception as e:
        log.error("Ошибка запроса к Prometheus (%s): %s", metric_name, e)
        return None

# ── PostgreSQL ────────────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def wait_for_db(retries: int = 10, delay: int = 5):
    for attempt in range(1, retries + 1):
        try:
            conn = get_connection()
            conn.close()
            log.info("PostgreSQL доступен.")
            return
        except Exception as e:
            log.warning("PostgreSQL недоступен (попытка %d/%d): %s", attempt, retries, e)
            time.sleep(delay)
    raise RuntimeError("Не удалось подключиться к PostgreSQL.")

def insert_metrics(conn, row: dict):
    sql = """
        INSERT INTO metrics (collected_at, cpu_usage, ram_usage, disk_usage, net_rx_bytes, net_tx_bytes)
        VALUES (%(collected_at)s, %(cpu_usage)s, %(ram_usage)s, %(disk_usage)s, %(net_rx_bytes)s, %(net_tx_bytes)s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)
    conn.commit()

# ── Основной цикл ─────────────────────────────────────────────────────────────
def collect_once(conn):
    values = {name: query_prometheus(name, promql) for name, promql in QUERIES.items()}

    # Пропускаем итерацию если хотя бы CPU или RAM не получили значение
    if values["cpu_usage"] is None or values["ram_usage"] is None:
        log.warning("Пропускаем запись: ключевые метрики недоступны.")
        return

    row = {
        "collected_at": datetime.now(timezone.utc),
        "cpu_usage":    values["cpu_usage"],
        "ram_usage":    values["ram_usage"],
        "disk_usage":   values.get("disk_usage") or 0.0,
        "net_rx_bytes": values.get("net_rx_bytes") or 0.0,
        "net_tx_bytes": values.get("net_tx_bytes") or 0.0,
    }
    insert_metrics(conn, row)
    log.info(
        "Записано → cpu=%.2f%% ram=%.2f%% disk=%.2f%%",
        row["cpu_usage"], row["ram_usage"], row["disk_usage"]
    )

def main():
    log.info("Collector запущен. Интервал: %d сек.", COLLECT_INTERVAL)
    wait_for_db()
    conn = get_connection()
    try:
        while True:
            collect_once(conn)
            time.sleep(COLLECT_INTERVAL)
    except KeyboardInterrupt:
        log.info("Остановка collector.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
