import os
import mlflow
import pandas as pd
import psycopg2

from celery_app import celery_app

# --- 1. GLOBAL MODEL HOLDER ---
# Keep it None at startup to prevent crashes if MLflow is empty
model = None


def _predict_batch(loaded_model, texts: list):
    texts_list = ["" if t is None else str(t) for t in texts]

    # Most sklearn pipelines (TF-IDF etc.) expect a 1D iterable of strings.
    # Passing a DataFrame makes sklearn treat it as an iterable of column names,
    # producing a single prediction and causing index errors for batches.
    last_exc = None

    for candidate in (
        texts_list,
        pd.Series(texts_list),
        pd.DataFrame({"text": texts_list}),
        pd.DataFrame({"description": texts_list}),
    ):
        try:
            preds = loaded_model.predict(candidate)
            # normalize to a plain list
            if hasattr(preds, "tolist"):
                preds = preds.tolist()
            else:
                preds = list(preds)

            if len(preds) != len(texts_list):
                raise ValueError(
                    f"Prediction length mismatch: got {len(preds)} preds for {len(texts_list)} texts"
                )
            return preds
        except Exception as e:
            last_exc = e
            continue

    raise last_exc

# --- 2. BACKGROUND TASK ---
@celery_app.task(bind=True, name="predict_genre_task")
def predict_genre_task(self, texts: list):
    """
    Processes a batch of text, predicts genre, and logs results to PostgreSQL.
    """
    global model
    
    # --- 4. LAZY LOAD MODEL ---
    # Only try to load the model when the first task actually arrives
    if model is None:
        try:
            MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SVC_model")
            MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            # Pulls the absolute latest version registered
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
            print(f"Successfully loaded model: {MODEL_NAME}")
        except Exception as e:
            print(f"Model Loading Error: {e}")
            raise e

    # Perform prediction on the whole batch for efficiency
    predictions = _predict_batch(model, texts)
    
    # Get the unique Task ID for this specific execution
    task_id = self.request.id 
    
    all_results = []

    # --- 5. DATABASE LOGGING ---
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "db"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        cur = conn.cursor()

        for i, text in enumerate(texts):
            raw_pred = predictions[i]

            # Convert to numeric label (1 for Fiction, 0 for Nonfiction)
            if isinstance(raw_pred, str):
                pred_norm = raw_pred.strip().lower()
                if pred_norm in {"fiction", "1", "true"}:
                    pred_id = 1
                elif pred_norm in {"nonfiction", "non-fiction", "0", "false"}:
                    pred_id = 0
                else:
                    pred_id = int(raw_pred)
            else:
                pred_id = int(raw_pred)

            genre = "Fiction" if pred_id == 1 else "Nonfiction"

            # Log to PostgreSQL including the task_id
            cur.execute(
                """INSERT INTO genre_predictions (input_text, prediction, label_id, task_id) 
                   VALUES (%s, %s, %s, %s)""",
                (text, genre, pred_id, task_id)
            )
            
            all_results.append({"prediction": genre, "label_id": pred_id})

        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print(f"Worker Database Error: {e}")
        raise e 

    return all_results