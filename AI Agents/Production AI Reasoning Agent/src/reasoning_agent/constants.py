"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"
PROMPTS_DIR = PROJECT_ROOT / "src" / "reasoning_agent" / "prompts"
BENCHMARK_DATA_PATH = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_prompts.jsonl"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
