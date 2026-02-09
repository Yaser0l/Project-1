# 📚 Book Genre Classification Pipeline

A full-stack MLOps project for training, registering, and serving a machine learning model to classify book descriptions as **Fiction** or **Nonfiction**.



## 🏗️ System Architecture

The project is built using a microservices architecture deployed on **Google Cloud Platform (GCP)** via **Coolify**:

* **FastAPI**: The production endpoint that receives text data and returns genre predictions.
* **MLflow**: An experiment tracking server and Model Registry that versions and stores our trained models.
* **PostgreSQL**: A relational database used as the backend for MLflow metadata and for logging production prediction results.
* **Docker Compose**: Orchestrates the containers, networking, and volumes for data persistence.

---

## 🚀 Getting Started

### 1. Local Development & Training
Training is performed on your local machine to leverage local resources while logging results to the remote server.

1.  **Configure Environment**: Update your local `.env` file with the VM Public IP:
    ```env
    POSTGRES_USER= YOUR USERNAME
    POSTGRES_PASSWORD= YOUR PASSWORD
    POSTGRES_DB= YOUR DB NAME
    POSTGRES_HOST= YOUR HOSTNAME
    POSTGRES_PORT=5432
    MLFLOW_TRACKING_URI= YOUR MLFLOW URI
    ```
    Note: make sure your .env is consistant with you notebook and scripts
2.  **Run Training**: Execute your Jupyter Notebook or Python script. This will:
    * Train a Support Vector Classifier (SVC).
    * Log metrics (Accuracy, Precision, Recall) to the remote MLflow.
    * **Register** the model as `SVC_model`.

### 2. Model Serving (FastAPI)
Once registered, the API (running on the VM) pulls the model directly from the internal MLflow service.

* **Interactive API Docs**: `http://YOUR VM IP:8001/docs`
* **Health Check**: `http://YOUR VM IP:8001/health`

---

## 🛠️ Infrastructure Configuration

| Service | Internal Port | Public Port | Access |
| :--- | :--- | :--- | :--- |
| **FastAPI** | 8000 | 8001 | Public API Endpoint |
| **MLflow** | 5000 | 5001 | Web UI & Tracking |
| **PostgreSQL** | 5432 | 5432 | DBeaver / Database Access |

### Critical Environment Variables
These are managed in the Coolify Dashboard to ensure secure internal communication:
* `POSTGRES_USER= YOUR USERNAME`
  
* `POSTGRES_PASSWORD= YOUR PASSWORD`
  
* `POSTGRES_DB= YOUR DB NAME`
  
* `MLFLOW_TRACKING_URI=`: `http://mlflow:5000` (Internal Docker network).

---

## 💻 Usage Example

To get a prediction from the live server:
Open your FastAPI SwaggerUI and predict
or
```bash
curl -X 'POST' \
  '[http://34.67.228.231:8001/predict](http://34.67.228.231:8001/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "A young boy discovers he is a wizard and attends a magical boarding school.",
    "A detailed biography of a famous historical figure during the industrial revolution."
  ]
}'
