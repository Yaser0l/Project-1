import time

from locust import HttpUser, task, between

class BookApiUser(HttpUser):
    # A small wait prevents hammering /status and gives the worker time to process.
    wait_time = between(0.5, 1.5)

    # Polling configuration (end-to-end test)
    status_poll_interval_s = 1.0
    status_timeout_s = 30.0

    @task
    def predict_genre(self):
        # This matches your FastAPI 'BookBatchInput' schema
        payload = {
            "texts": [
                "A thriller about a secret society.A thriller about a secret society.A thriller about a secret society.A thriller about a secret society.",
                "The history of modern architecture.The history of modern architecture.The history of modern architecture.The history of modern architecture.",
                "A cooking guide for beginners.A cooking guide for beginners.A cooking guide for beginners.A cooking guide for beginners.",
                "Space exploration in the year 3000.Space exploration in the year 3000.Space exploration in the year 3000.Space exploration in the year 3000."
            ]
        }
        
        # 1) Enqueue prediction (fast path)
        with self.client.post("/predict", json=payload, catch_response=True, name="/predict") as response:
            if response.status_code != 200:
                response.failure(f"/predict returned {response.status_code}")
                return

            try:
                task_id = response.json().get("task_id")
            except Exception as e:
                response.failure(f"/predict invalid JSON: {e}")
                return

            if not task_id:
                response.failure("/predict response missing task_id")
                return

            response.success()

        # 2) Poll status until done (slow path)
        # Uses cookie-based convenience endpoint: GET /status
        deadline = time.time() + self.status_timeout_s
        while time.time() < deadline:
            with self.client.get("/status", catch_response=True, name="/status") as status_resp:
                if status_resp.status_code != 200:
                    status_resp.failure(f"/status returned {status_resp.status_code}")
                    return

                try:
                    body = status_resp.json()
                except Exception as e:
                    status_resp.failure(f"/status invalid JSON: {e}")
                    return

                status = body.get("status")
                if status in {"PENDING", "STARTED"}:
                    status_resp.success()
                elif status == "SUCCESS":
                    result = body.get("result")
                    if not isinstance(result, list):
                        status_resp.failure("/status SUCCESS but result is not a list")
                        return
                    status_resp.success()
                    return
                else:
                    # FAILURE or unexpected state
                    status_resp.failure(f"Task ended with status={status}, error={body.get('error')}")
                    return

            time.sleep(self.status_poll_interval_s)

        # Timeout
        self.environment.events.request.fire(
            request_type="GET",
            name="/status",
            response_time=self.status_timeout_s * 1000,
            response_length=0,
            exception=TimeoutError(f"Timed out waiting for task {task_id}"),
        )