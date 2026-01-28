from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
import os

# ==========================================
# 1. NETWORK MAPPING (Connection String)
# ==========================================
# We use "localhost" because we are running this on Windows, outside the container.

from dotenv import load_dotenv
env_path = Path(__file__).resolve().parents[1] / ".env"  # project root/.env
load_dotenv(dotenv_path=env_path)

DB_HOST = "localhost"
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_PORT = "5432"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# ==========================================
# 2. FILE PATH MAPPING
# ==========================================
# 
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data\\goodreads_data.csv')

if not os.path.exists(csv_path):
    print(f"❌ ERROR: File not found at: {csv_path}")
    exit()

print(f"✅ Found file: {csv_path}")
df = pd.read_csv(csv_path)

# ==========================================
# 3. COLUMN MAPPING (Schema)
# ==========================================
# Left Side: The name inside the CSV file
# Right Side: The name inside your SQL Table
column_map = {
    'Unnamed: 0':  'id',
    'Book':        'book',
    'Author':      'author',
    'Description': 'description',
    'Genres':      'genres',
    'Avg_Rating':  'avg_rating',
    'Num_Ratings': 'num_ratings_raw',  # Mapping "Num_Ratings" to "num_ratings_raw"
    'URL':         'url',
}

# Apply the mapping
df = df.rename(columns=column_map)

# Only keep columns that actually exist in the SQL table
expected_columns = ['id', 'book', 'author', 'description', 'genres', 'avg_rating', 'num_ratings_raw', 'url']
df_final = df[expected_columns]

# ==========================================
# 4. EXECUTION (Load Data)
# ==========================================
try:
    inspector = inspect(engine)
    if not inspector.has_table('raw_data'):
        print("❌ ERROR: Table 'raw_data' does not exist.")
        raise SystemExit(1)

    # Insert rows; ignore duplicates by primary key (id)
    metadata = MetaData()
    raw_table = Table('raw_data', metadata, autoload_with=engine)

    table_cols = {c.name for c in raw_table.columns}
    insert_cols = [c for c in df_final.columns if c in table_cols]
    if 'id' not in insert_cols:
        raise ValueError("Table 'raw_data' must have an 'id' column.")

    inserted_total = 0
    chunk_size = 1000
    for start in range(0, len(df_final), chunk_size):
        chunk = df_final.iloc[start:start + chunk_size][insert_cols]
        rows = chunk.to_dict(orient='records')
        if not rows:
            continue

        stmt = (
            pg_insert(raw_table)
            .values(rows)
            .on_conflict_do_nothing(index_elements=['id'])
        )

        with engine.begin() as conn:
            result = conn.execute(stmt)
            if result.rowcount is not None:
                inserted_total += result.rowcount
    if inserted_total == 0:
        print("⚠️ No new rows were inserted; all data may already exist in 'raw_data'.") 
    else:
        print(f"✅ SUCCESS: Loaded into 'raw_data' (inserted {inserted_total} new rows; duplicates ignored).")
except Exception as e:
    print(f"❌ ERROR: {e}")