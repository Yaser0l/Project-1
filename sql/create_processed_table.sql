CREATE TABLE IF NOT EXISTS processed_data (
    -- ML-ready processed dataset
    id BIGINT   PRIMARY KEY,
    genre_group TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_processed_data_genre_group ON processed_data (genre_group);