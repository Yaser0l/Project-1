import sys
import re
import os
import joblib
import psycopg2
import logging 
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# --- 0. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # This sends logs to terminal
    ]
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION & LIMITS ---
load_dotenv()

API_MAX_INPUT = 100    
CHUNKING_SIZE = 20     

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

# --- 2. PREPROCESSING & PIPELINE FIX ---
lemmatizer = WordNetLemmatizer()
stop_words = set(['the', 'is', 'at', 'which', 'on']) 

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def pipeline_preprocess(text_series):
    if isinstance(text_series, str):
        text_series = [text_series]
    return [preprocess_text(text) for text in text_series]

import __main__
__main__.pipeline_preprocess = pipeline_preprocess
sys.modules['pipeline_preprocess'] = pipeline_preprocess 

# --- 3. INITIALIZE API & MODEL ---
app = FastAPI(title="Genre Classifier API - Logging Edition")
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "notebooks", "SVC_pipeline.joblib")
model = joblib.load(model_path)

class BookBatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=API_MAX_INPUT)

# --- 4. DATABASE LOGIC ---
def save_prediction(text, genre, label_id):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS genre_predictions (
                id SERIAL PRIMARY KEY,
                input_text TEXT,
                prediction VARCHAR(20),
                label_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute(
            "INSERT INTO genre_predictions (input_text, prediction, label_id) VALUES (%s, %s, %s)",
            (text, genre, label_id)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error while saving prediction: {e}")

# --- 5. CHUNKING HELPER ---
def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# --- 6. ENDPOINTS ---
@app.post("/predict")
def predict_genre(data: BookBatchInput):
    total_texts = len(data.texts)
    logger.info(f"Received request with {total_texts} texts. Max input: {API_MAX_INPUT}")
    
    try:
        all_results = []
        chunks = list(get_chunks(data.texts, CHUNKING_SIZE))
        
        for index, chunk in enumerate(chunks):
            current_chunk_num = index + 1
            logger.info(f"Processing chunk {current_chunk_num}/{len(chunks)} (Size: {len(chunk)})")
            
            # Predict the entire chunk
            predictions = model.predict(chunk)
            
            for i, text in enumerate(chunk):
                pred_id = int(predictions[i])
                genre = "Fiction" if pred_id == 1 else "Nonfiction"
                
                # Save to Postgres
                save_prediction(text, genre, pred_id)
                
                all_results.append({
                    "prediction": genre,
                    "label_id": pred_id
                })
            
            logger.info(f"Successfully finished chunk {current_chunk_num}")

        logger.info(f"Request completed. Total processed: {len(all_results)}")
        
        return {
            "total_processed": len(all_results),
            "chunks_handled": len(chunks),
            "results": all_results
        }
        
    except Exception as e:
        logger.exception("A critical error occurred during processing")
        raise HTTPException(status_code=500, detail="Internal processing error")