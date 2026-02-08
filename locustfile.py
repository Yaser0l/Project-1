import random
from locust import HttpUser, task, between

class BookApiUser(HttpUser):
    # wait_time = 

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
        
        # self.client is like 'requests' but tracks stats for Locust
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")