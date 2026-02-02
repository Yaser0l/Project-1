import sys
import re
import os
import joblib
import psycopg2
import logging 
import mlflow 
import mlflow.sklearn 
from typing import List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# --- 0. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION & LIMITS ---
load_dotenv()

API_MAX_INPUT = 100    
CHUNKING_SIZE = 20     

POSTGRES_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

# --- 2. MLFLOW INITIALIZATION ---
mlflow.set_tracking_uri(POSTGRES_URL)
mlflow.set_experiment("Genre_Classifier_TCC")

# --- 3. INPUT MODELS (FIXED: Defined BEFORE the endpoint) ---
class BookBatchInput(BaseModel):
    """
    Pydantic model to validate the incoming batch of texts.
    Ensures the list contains between 1 and 100 items.
    """
    texts: List[str] = Field(..., min_length=1, max_length=API_MAX_INPUT)

# --- 4. PREPROCESSING & PIPELINE FIX ---
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

# --- 5. INITIALIZE API & MODEL ---
app = FastAPI(title="Genre Classifier API - Final Edition")
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "notebooks", "SVC_pipeline.joblib")
model = joblib.load(model_path)

# Log Model as an Artifact on startup
try:
    with mlflow.start_run(run_name="App_Startup_Artifact"):
        mlflow.sklearn.log_model(sk_model=model, artifact_path="svc_model")
        logger.info("✅ Artifact successfully created and stored in MLflow.")
except Exception as e:
    logger.warning(f"⚠️ Could not create artifact: {e}")

# --- 6. DATABASE & CHUNKING HELPERS ---
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
        logger.error(f"❌ Database error: {e}")

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# --- 7. ENDPOINTS ---
@app.post("/predict")
def predict_genre(data: BookBatchInput):
    total_texts = len(data.texts)
    logger.info(f"🚀 Received request with {total_texts} texts.")
    
    with mlflow.start_run(run_name=f"Predict_Batch_{datetime.now().strftime('%H%M%S')}"):
        try:
            all_results = []
            fiction_count = 0
            nonfiction_count = 0
            
            mlflow.log_param("batch_size", total_texts)
            chunks = list(get_chunks(data.texts, CHUNKING_SIZE))
            
            for index, chunk in enumerate(chunks):
                logger.info(f"📦 Processing chunk {index+1}/{len(chunks)}")
                predictions = model.predict(chunk)
                
                for i, text in enumerate(chunk):
                    pred_id = int(predictions[i])
                    genre = "Fiction" if pred_id == 1 else "Nonfiction"
                    
                    if pred_id == 1: fiction_count += 1
                    else: nonfiction_count += 1

                    save_prediction(text, genre, pred_id)
                    all_results.append({"prediction": genre, "label_id": pred_id})
            
            mlflow.log_metric("fiction_count", fiction_count)
            mlflow.log_metric("nonfiction_count", nonfiction_count)
            
            return {
                "total_processed": len(all_results),
                "chunks_handled": len(chunks),
                "results": all_results
            }
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            logger.exception("A critical error occurred")
            raise HTTPException(status_code=500, detail="Internal processing error")