CREATE TABLE IF NOT EXISTS genre_predictions (
    id          SERIAL PRIMARY KEY,
    input_text  TEXT NOT NULL,
    prediction  TEXT NOT NULL,
    label_id    INTEGER NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_genre_predictions_created_at
    ON genre_predictions (created_at);

CREATE INDEX IF NOT EXISTS idx_genre_predictions_label_id
    ON genre_predictions (label_id);