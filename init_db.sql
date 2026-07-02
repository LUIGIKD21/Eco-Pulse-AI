-- Drops existing conflict schemas to guarantee fresh MVP data layer migrations
DROP TABLE IF EXISTS prediction_logs;

CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    actual_temp NUMERIC(5, 2) NOT NULL,
    sentiment_score NUMERIC(3, 2) NOT NULL,
    predicted_kwh NUMERIC(10, 2) NOT NULL,
    confidence_interval NUMERIC(3, 2) NOT NULL
);

-- Crucial index execution to scale fetch metrics within <500ms parameters [cite: 63]
CREATE INDEX idx_timestamp_desc ON prediction_logs (timestamp DESC);