import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


def _find_free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError:
        pytest.skip("Socket bind blocked in current sandbox environment")


@pytest.fixture(scope="module")
def live_api_base_url() -> str:
    root = Path(__file__).resolve().parents[1]
    port = _find_free_port()
    env = os.environ.copy()
    env["ML_MODEL_PATH"] = str(root / "models" / "iris_model.pkl")
    env["ML_REGISTRY_PATH"] = str(root / "models" / "registry.json")
    env["ML_BACKGROUND_DATA_PATH"] = str(root / "models" / "background_data.npy")
    env["ML_API_HOST"] = "127.0.0.1"
    env["ML_API_PORT"] = str(port)
    env["MPLCONFIGDIR"] = str(root / ".mplconfig")

    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(40):
            try:
                response = requests.get(f"{base_url}/health", timeout=1.5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("Timed out waiting for live API server startup")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_live_health_endpoint_requests(live_api_base_url: str) -> None:
    response = requests.get(f"{live_api_base_url}/health", timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert "model_loaded" in body


def test_live_predict_endpoint_requests(live_api_base_url: str) -> None:
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = requests.post(f"{live_api_base_url}/predict", json=payload, timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "species" in body
