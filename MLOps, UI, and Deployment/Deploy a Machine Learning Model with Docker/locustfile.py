"""Locust load test scenario for the containerized ML API."""

from __future__ import annotations

from locust import HttpUser, between, task

SAMPLE_INPUT = {
    "MedInc": 8.3,
    "HouseAge": 42.0,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 230.0,
    "AveOccup": 3.2,
    "Latitude": 37.9,
    "Longitude": -122.2,
}


class MLAPIUser(HttpUser):
    """Simulate realistic mixed traffic against inference API."""

    wait_time = between(0.05, 0.5)

    @task(1)
    def health(self) -> None:
        self.client.get("/health")

    @task(2)
    def single_prediction(self) -> None:
        self.client.post("/predict", json=SAMPLE_INPUT)

    @task(1)
    def batch_prediction(self) -> None:
        self.client.post("/predict-batch", json={"instances": [SAMPLE_INPUT, SAMPLE_INPUT]})

    @task(1)
    def explanation(self) -> None:
        self.client.post("/explain", json={"input": SAMPLE_INPUT})

