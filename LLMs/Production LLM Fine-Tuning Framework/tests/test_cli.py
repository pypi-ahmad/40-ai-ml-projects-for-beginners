import os
from pathlib import Path

import yaml

from llmft.cli import main


def test_cli_env_and_data(tmp_path) -> None:
    os.environ["LLMFT_TRANSFORMERS_MODEL"] = "sshleifer/tiny-gpt2"
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    payload["runtime"]["artifacts_dir"] = str(tmp_path / "artifacts")
    payload["runtime"]["cache_dir"] = str(tmp_path / "cache")
    payload["data"]["datasets"] = ["alpaca_cleaned"]
    payload["data"]["max_samples_per_dataset"] = 8
    payload["inference"]["vllm_host"] = "http://127.0.0.1:9"
    payload["inference"]["ollama_host"] = "http://127.0.0.1:9"
    payload["inference"]["benchmark_backends"] = ["transformers"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    main(["--config", str(config_path), "env", "validate"])
    main(["--config", str(config_path), "data", "build"])
