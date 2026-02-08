import sys
import os
import psycopg2
import logging
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --- 2. CONFIGURATION ---
load_dotenv()

API_MAX_INPUT = 100
CHUNKING_SIZE = 20

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}


# --- 3. INITIALIZE API & MODEL ---
app = FastAPI(title="Genre Classifier API")

model: Optional[Any] = None
last_model_load_error: Optional[str] = None


def _build_model_uri(model_name: str, model_stage: Optional[str], model_version: Optional[str]) -> str:
    if model_stage and model_stage.strip():
        return f"models:/{model_name}/{model_stage.strip()}"
    if model_version and str(model_version).strip():
        return f"models:/{model_name}/{str(model_version).strip()}"
    raise ValueError("No model stage or version provided")


def _get_latest_model_version(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No versions found for registered model '{model_name}'. "
            "Register a model version in MLflow first."
        )
    latest = max(int(v.version) for v in versions)
    return str(latest)


def load_model_from_mlflow() -> Optional[Any]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv("MLFLOW_MODEL_NAME", "SVC_model")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE")
    model_version = os.getenv("MLFLOW_MODEL_VERSION")

    client = MlflowClient(tracking_uri=tracking_uri)

    if not (model_stage and model_stage.strip()) and not (model_version and model_version.strip()):
        model_version = _get_latest_model_version(client, model_name)

    model_uri = _build_model_uri(model_name=model_name, model_stage=model_stage, model_version=model_version)

    logger.info(f"Loading model from MLflow: tracking_uri={tracking_uri}, model_uri={model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


@app.on_event("startup")
def load_model_on_startup() -> None:
    global model
    global last_model_load_error
    try:
        model = load_model_from_mlflow()
        last_model_load_error = None
        logger.info("Model loaded successfully")
    except Exception as e:
        # Keep API up even if MLflow is temporarily unavailable.
        logger.exception("Could not load model from MLflow on startup")
        model = None
        last_model_load_error = f"Startup load failed: {type(e).__name__}: {e}"


class BookBatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=API_MAX_INPUT)


# --- 4. DATABASE LOGIC ---
def save_prediction(text: str, genre: str, label_id: int) -> None:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO genre_predictions (input_text, prediction, label_id) VALUES (%s, %s, %s)",
            (text, genre, label_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error while saving prediction: {e}")


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _prediction_to_label(prediction) -> int:
    """Normalize model output to 0/1 label."""
    if prediction is None:
        return 0
    if isinstance(prediction, (bool, int)):
        return int(prediction)
    s = str(prediction).strip().lower()
    if s in {"1", "fiction", "true"}:
        return 1
    if s in {"0", "nonfiction", "non-fiction", "false"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


# --- 5. ENDPOINTS ---
@app.post("/predict")
def predict_genre(data: BookBatchInput):
    global model
    global last_model_load_error

    # If MLflow wasn't ready at startup, try once more on demand.
    if model is None:
        try:
            model = load_model_from_mlflow()
            last_model_load_error = None
            logger.info("Model loaded successfully (on-demand)")
        except Exception as e:
            logger.exception("Model not initialized (MLflow load failed)")
            last_model_load_error = f"On-demand load failed: {type(e).__name__}: {e}"
            raise HTTPException(
                status_code=503,
                detail="Model not initialized. Ensure the model is registered in MLflow and try again.",
            )

    total_texts = len(data.texts)
    logger.info(f"Received request with {total_texts} texts.")

    try:
        all_results = []
        chunks = list(get_chunks(data.texts, CHUNKING_SIZE))

        for index, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {index + 1}/{len(chunks)}")

            predictions = model.predict(chunk)

            for i, text in enumerate(chunk):
                pred_id = _prediction_to_label(predictions[i])
                genre = "Fiction" if pred_id == 1 else "Nonfiction"

                save_prediction(text, genre, pred_id)

                all_results.append({"prediction": genre, "label_id": pred_id})

        logger.info(f"Request completed. Total processed: {len(all_results)}")
        return {"total_processed": len(all_results), "results": all_results}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Critical error during prediction")
        raise HTTPException(status_code=500, detail="Internal processing error")


@app.get("/health")
def health() -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "SVC_model")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE")
    model_version = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mlflow": {
            "tracking_uri": tracking_uri,
            "model_name": model_name,
            "model_stage": model_stage,
            "model_version": model_version,
        },
        "last_model_load_error": last_model_load_error,
    }
