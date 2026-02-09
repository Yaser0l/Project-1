import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field
from celery.result import AsyncResult

from celery_app import celery_app

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
API_MAX_INPUT = 100
app = FastAPI(title="Genre Classifier API (Asynchronous)")

class BookBatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=API_MAX_INPUT)

# --- 3. ENDPOINTS ---

@app.post("/predict")
async def predict(data: BookBatchInput, response: Response):
    """
    Hands the batch to Redis and returns a receipt immediately.
    """
    try:
        total_texts = len(data.texts)
        logger.info(f"Received batch of {total_texts} texts for background processing.")

        # Send the task to the queue (Redis). Non-blocking.
        task = celery_app.send_task("predict_genre_task", args=[data.texts])

        # Convenience: store the last task id in a cookie so you can call GET /status
        # without manually copying the task id.
        response.set_cookie(
            key="last_task_id",
            value=task.id,
            httponly=True,
            samesite="lax",
        )

        return {
            "message": "Task received and queued.",
            "task_id": task.id,
            "status_url": f"/status/{task.id}"
        }
    except Exception as e:
        logger.error(f"Failed to queue task: {e}")
        raise HTTPException(status_code=500, detail="Internal queueing error. Is Redis running?")

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Checks if the background worker has finished the job.
    """
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        
        # Possible statuses: PENDING, STARTED, SUCCESS, FAILURE
        response = {
            "task_id": task_id,
            "status": task_result.status,
            "result": None
        }

        if task_result.ready():
            if task_result.status == "SUCCESS":
                response["result"] = task_result.result
            else:
                # Provides the error traceback if the task failed
                response["error"] = str(task_result.info)
                
        return response
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found or expired.")


@app.get("/status")
async def get_last_task_status(request: Request):
    """
    Convenience endpoint that returns the status for the most recent task id
    created by this client (stored in the `last_task_id` cookie).
    """
    task_id = request.cookies.get("last_task_id")
    if not task_id:
        raise HTTPException(
            status_code=400,
            detail="No last task id found. Call POST /predict first or use GET /status/{task_id}.",
        )

    task_result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None,
    }

    if task_result.ready():
        if task_result.status == "SUCCESS":
            response["result"] = task_result.result
        else:
            response["error"] = str(task_result.info)

    return response

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "asynchronous_queue"}