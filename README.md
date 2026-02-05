# Docker EDA pipeline (raw → EDA → processed)

Pipeline steps:
1) Load raw files into Postgres (Docker)
2) Generate an EDA report in Docker by reading raw table
3) Write processed data back into Postgres in a different table

## Prereqs
- Docker Desktop
- `docker compose` available

## Quickstart
1) Create env file:

```bash
copy .env.example .env
```

2) Put one or more CSVs into `data/`.

3) Start database:

```bash
docker compose up -d
```

4) Ingest raw CSVs into Postgres:

```bash
docker compose --profile jobs run --rm ingest_raw
```

5) Generate EDA report + write processed table:

```bash
docker compose --profile jobs run --rm eda_process
```

6) Open report:
- `reports/eda_report.html`

## Notes
- Raw data loads into `${RAW_TABLE}` (default `raw_data`).
- Processed data writes into `${PROCESSED_TABLE}` (default `processed_data`).
- Ingestion is idempotent per file using a SHA-256 hash stored in `ingestion_metadata`.