"""Train and serialize FastAPI model artifacts."""

from __future__ import annotations

import json

from ml_api.core.config import get_settings
from ml_api.training.pipeline import train_and_serialize


def main() -> int:
    settings = get_settings()
    outputs = train_and_serialize(settings)

    print("Training complete")
    print(json.dumps(
        {
            "model_name": outputs.metadata.model_name,
            "model_type": outputs.metadata.model_type,
            "validation_rmse": outputs.metadata.validation_rmse,
            "test_rmse": outputs.metadata.test_rmse,
            "benchmark_csv": str(outputs.benchmark_csv),
            "benchmark_json": str(outputs.benchmark_json),
            "automl_json": str(outputs.automl_json),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
