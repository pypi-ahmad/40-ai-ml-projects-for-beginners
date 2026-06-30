import logging
import tempfile

from ml_package.logging_config import PredictionLogger, setup_logging


class TestLogging:
    def test_setup_logging_returns_logger(self):
        logger = setup_logging("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_prediction_logger(self):
        logger = PredictionLogger("pred_test")
        logger.log_prediction(
            features={"sepal_length": 5.1, "sepal_width": 3.5},
            prediction=0,
            confidence=0.95,
            latency_ms=2.5,
        )
        assert logger.prediction_count() >= 1

    def test_prediction_logger_get_stats(self):
        logger = PredictionLogger("stats_test")
        logger.log_prediction(
            features={"sepal_length": 5.1},
            prediction=0,
            confidence=0.95,
            latency_ms=2.5,
        )
        stats = logger.get_stats()
        assert stats["total_predictions"] >= 1
