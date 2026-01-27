import pandas as pd
from sqlalchemy import create_engine
import os

# ==========================================
# 1. NETWORK MAPPING (Connection String)
# ==========================================
# We use "localhost" because we are running this on Windows, outside the container.
DB_HOST = "localhost"
DB_NAME = "mldb"
DB_USER = "mluser"
DB_PASS = "mlpass"
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
    'URL':         'url'
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
    df_final.to_sql(
        name='raw_data',     # Target SQL Table Name
        con=engine,
        if_exists='append',  # Add to table, don't replace it
        index=False,         # We already have an 'id' column, so don't make a new index
        chunksize=1000       # Upload 1000 rows at a time
    )
    print("✅ SUCCESS: Data successfully mapped and loaded into PostgreSQL!")
except Exception as e:
    print(f"❌ ERROR: {e}")