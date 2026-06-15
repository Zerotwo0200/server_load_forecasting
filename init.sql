-- Таблица с историческими метриками (сырые данные)
CREATE TABLE IF NOT EXISTS metrics (
    id          BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cpu_usage   DOUBLE PRECISION NOT NULL,
    ram_usage   DOUBLE PRECISION NOT NULL,
    disk_usage  DOUBLE PRECISION NOT NULL,
    net_rx_bytes DOUBLE PRECISION,
    net_tx_bytes DOUBLE PRECISION
);

-- Таблица с прогнозами от ML-модели
CREATE TABLE IF NOT EXISTS predictions (
    id           BIGSERIAL PRIMARY KEY,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_time  TIMESTAMPTZ NOT NULL,
    metric_name  VARCHAR(64) NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(32)
);

-- Таблица рекомендаций (LIVE решения)
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    predicted_cpu FLOAT,
    predicted_ram FLOAT,
    recommendation TEXT,
    status TEXT,
    message TEXT
);

-- Таблица истории реальных событий масштабирования
CREATE TABLE IF NOT EXISTS scaling_history (
    id             SERIAL PRIMARY KEY,
    event_time     TIMESTAMPTZ DEFAULT NOW(),
    action         VARCHAR(50) NOT NULL,
    old_size       INT NOT NULL,
    new_size       INT NOT NULL,
    trigger_reason TEXT NOT NULL
);

-- Индексы для скорости
CREATE INDEX IF NOT EXISTS idx_metrics_collected_at ON metrics (collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_target_time ON predictions (target_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_metric ON predictions (metric_name, target_time DESC);
CREATE INDEX IF NOT EXISTS idx_scaling_history_time ON scaling_history (event_time DESC);

-- Базовая запись для старта алгоритма
INSERT INTO scaling_history (event_time, action, old_size, new_size, trigger_reason)
SELECT NOW(), 'INITIALIZE', 2, 2, 'Система запущена. Базовый размер инфраструктуры: 2 сервера.'
WHERE NOT EXISTS (SELECT 1 FROM scaling_history);

CREATE TABLE IF NOT EXISTS alert_configs (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(64) NOT NULL,
    steps INT NOT NULL,
    threshold FLOAT NOT NULL,
    condition VARCHAR(10) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE
);

