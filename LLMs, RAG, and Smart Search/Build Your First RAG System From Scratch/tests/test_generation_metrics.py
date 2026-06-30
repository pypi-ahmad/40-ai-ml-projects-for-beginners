from __future__ import annotations

from rag_system.metrics import compute_generation_metrics


class FakeMetric:
    def __init__(self, name: str, calls: dict[str, dict]) -> None:
        self.name = name
        self.calls = calls

    def compute(self, **kwargs):
        self.calls[self.name] = kwargs
        if self.name == "bleu":
            return {"bleu": 0.11}
        if self.name == "rouge":
            return {"rouge1": 0.22, "rougeL": 0.18}
        if self.name == "meteor":
            return {"meteor": 0.33}
        if self.name == "bertscore":
            return {"f1": [0.66, 0.77]}
        raise AssertionError(f"Unexpected metric name: {self.name}")


def test_generation_metrics_use_expected_reference_shapes(monkeypatch) -> None:
    calls: dict[str, dict] = {}

    def _fake_load_metric(name: str):
        return FakeMetric(name=name, calls=calls)

    monkeypatch.setattr("rag_system.metrics._load_metric", _fake_load_metric)

    summary = compute_generation_metrics(
        predictions=["Alpha answer", "Beta answer"],
        references=["Alpha answer", "Beta gold"],
    )

    bleu_refs = calls["bleu"]["references"]
    assert bleu_refs == [["Alpha answer"], ["Beta gold"]]
    assert calls["rouge"]["references"] == ["Alpha answer", "Beta gold"]
    assert calls["meteor"]["references"] == ["Alpha answer", "Beta gold"]
    assert calls["bertscore"]["references"] == ["Alpha answer", "Beta gold"]

    assert summary.exact_match == 0.5
    assert summary.bleu == 0.11
    assert summary.rouge1 == 0.22
    assert summary.rougeL == 0.18
    assert summary.meteor == 0.33
    assert abs(summary.bertscore_f1 - 0.715) < 1e-9
    assert summary.num_examples == 2
