CREATE TABLE IF NOT EXISTS raw_data(
    -- Primary Key for the Pandas index
    id    INTEGER PRIMARY KEY,
    
    book            TEXT,
    author          TEXT,
    description     TEXT,
    genres          TEXT,
    avg_rating      NUMERIC(3, 2),
    -- Keep as TEXT for now to handle commas (e.g., "5,691,311")
    num_ratings_raw TEXT,
    url             TEXT,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Helpful indexes for common lookups
CREATE INDEX IF NOT EXISTS ix_raw_data_book ON raw_data (book);
CREATE INDEX IF NOT EXISTS ix_raw_data_genres ON raw_data (genres);

-- Get-Content sql\create_raw_table.sql | docker exec -i ml_postgres sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB'
-- docker exec -i ml_postgres psql -U mluser -d mldb -f /sql/create_raw_table.sql