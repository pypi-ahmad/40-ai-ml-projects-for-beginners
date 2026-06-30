from llmft.config.schemas import DataConfig
from llmft.data import DatasetPipeline


def test_data_pipeline_builds_bundle(tmp_path) -> None:
    pipeline = DatasetPipeline(cache_dir=tmp_path / "cache", seed=123)
    config = DataConfig(datasets=["alpaca_cleaned"], max_samples_per_dataset=12, validation_ratio=0.2)

    bundle = pipeline.build(config, artifacts_dir=tmp_path / "artifacts")

    assert bundle.stats.rows_total > 0
    assert bundle.stats.rows_after_filter > 0
    assert bundle.stats.rows_train > 0
    assert bundle.manifest_path.exists()
    assert bundle.version_id.startswith("dataset-")
