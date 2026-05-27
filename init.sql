-- Таблица с историческими метриками (сырые данные от collector)
CREATE TABLE IF NOT EXISTS metrics (
    id          BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cpu_usage   DOUBLE PRECISION NOT NULL,  -- %
    ram_usage   DOUBLE PRECISION NOT NULL,  -- %
    disk_usage  DOUBLE PRECISION NOT NULL,  -- %
    net_rx_bytes DOUBLE PRECISION,          -- bytes/sec
    net_tx_bytes DOUBLE PRECISION           -- bytes/sec
);

-- Таблица с прогнозами от ML-модели
CREATE TABLE IF NOT EXISTS predictions (
    id           BIGSERIAL PRIMARY KEY,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_time  TIMESTAMPTZ NOT NULL,      -- на какой момент прогноз
    metric_name  VARCHAR(64) NOT NULL,      -- cpu/ram/disk
    predicted_value DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    predicted_cpu FLOAT,
    predicted_ram FLOAT,
    recommendation TEXT,
    status TEXT,
    message TEXT
);

-- Индексы для быстрых выборок по времени
CREATE INDEX IF NOT EXISTS idx_metrics_collected_at ON metrics (collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_target_time ON predictions (target_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_metric ON predictions (metric_name, target_time DESC);
