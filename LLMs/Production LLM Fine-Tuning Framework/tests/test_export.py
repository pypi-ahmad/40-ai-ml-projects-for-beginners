from llmft.config.schemas import ExportConfig
from llmft.export import ExportManager
from llmft.training.types import TrainingReport


def test_export_manager_writes_expected_artifacts(tmp_path) -> None:
    manager = ExportManager(ExportConfig(export_dir=str(tmp_path / "exports")))
    report = TrainingReport(
        run_id="run-1",
        model_alias="llama3_8b",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        peft_method="qlora",
        train_samples=100,
        validation_samples=10,
        steps=30,
        train_loss=0.2,
        eval_loss=0.3,
        checkpoints_dir=tmp_path / "ckpt",
        used_real_stack=False,
    )

    summary = manager.run(report)
    assert summary.exists()
    assert (summary.parent / "adapter.safetensors").exists()
    assert (summary.parent / "Modelfile").exists()
