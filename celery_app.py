import os
from celery import Celery

def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default

def make_celery_app() -> Celery:
    # Use the internal IP 10.128.0.2 in your GCP environment variables
    broker_url = _get_env("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend = _get_env("CELERY_RESULT_BACKEND", broker_url)

    app = Celery(
        "tasks",
        broker=broker_url,
        backend=result_backend,
    )

    # --- THE FIX: ADD THESE CONFIGURATIONS ---
    app.conf.update(
        # Connection Pooling: Reuses existing 'pipes' to Redis
        # Set this to 10 or 20 per API instance
        broker_pool_limit=10, 
        
        # Stability: Retries connection on startup instead of crashing
        broker_connection_retry_on_startup=True,
        
        # Health Checks: Keeps the connection alive under high load
        redis_backend_health_check_interval=30,
        
        # Performance: Don't wait for results to be acknowledged by the worker
        task_ignore_result=False, 
        
        broker_transport_options={'visibility_timeout': 3600}, # Prevents task duplication
        worker_prefetch_multiplier=1 # Ensures tasks are distributed evenly across workers
    )

    return app

celery_app = make_celery_app()