import json
import logging
import sys
from collections import deque
from datetime import datetime
from pathlib import Path


def setup_logging(
    name: str = "ml_package",
    log_dir: str = "logs",
    level: str = "INFO",
) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"ml_package_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_format = logging.Formatter(
            "%(levelname)-8s | %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class PredictionLogger:
    """Logs individual predictions with metadata for monitoring and audit."""

    def __init__(self, name: str, max_history: int = 5000):
        self.logger = logging.getLogger(name)
        self._predictions: deque[dict] = deque(maxlen=max_history)
        self._latency_total_ms = 0.0
        self._error_count = 0
        self._success_count = 0

    def log_prediction(
        self,
        features: dict,
        prediction: int,
        confidence: float,
        latency_ms: float,
        model_version: str = "1.0.0",
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": int(prediction),
            "confidence": confidence,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 3),
        }
        self._predictions.append(record)
        self._latency_total_ms += float(latency_ms)
        self._success_count += 1
        self.logger.info(f"Prediction: {json.dumps(record)}")

    def prediction_count(self) -> int:
        return self._success_count

    def get_stats(self) -> dict:
        avg_latency = (
            self._latency_total_ms / self._success_count if self._success_count > 0 else 0.0
        )
        return {
            "total_predictions": self._success_count,
            "total_errors": self._error_count,
            "average_latency_ms": round(avg_latency, 3),
        }

    def log_error(self, input_data: list, error: str) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "error": str(error),
        }
        self._predictions.append(record)
        self._error_count += 1
        self.logger.error(f"Prediction error: {json.dumps(record)}")

    def get_history(self) -> list[dict]:
        return list(self._predictions)
