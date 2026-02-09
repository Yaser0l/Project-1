import os

from celery import Celery


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def make_celery_app() -> Celery:
    broker_url = _get_env("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend = _get_env("CELERY_RESULT_BACKEND", broker_url)

    return Celery(
        "tasks",
        broker=broker_url,
        backend=result_backend,
    )


celery_app = make_celery_app()
