"""Task-specific adapters for zero/few-shot and hierarchical classification."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

from domain_llm_ft.models.registry import resolve_model_name


@dataclass
class TaskPrediction:
    label: str
    score: float


class ZeroShotClassifier:
    """NLI-based zero-shot classification adapter."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.pipe = pipeline("zero-shot-classification", model=resolve_model_name(model_name))

    def predict(self, text: str, candidate_labels: list[str]) -> TaskPrediction:
        output = self.pipe(text, candidate_labels)
        return TaskPrediction(label=output["labels"][0], score=float(output["scores"][0]))


class FewShotClassifier:
    """Few-shot prompt-based adapter for instruction models."""

    def __init__(self, model_name: str):
        self.pipe = pipeline("text-generation", model=resolve_model_name(model_name))

    def predict(self, text: str, examples: list[tuple[str, str]]) -> str:
        demonstration = "\n".join([f"Text: {inp}\nLabel: {label}" for inp, label in examples])
        prompt = f"{demonstration}\nText: {text}\nLabel:"
        output = self.pipe(prompt, max_new_tokens=16, do_sample=False)
        return output[0]["generated_text"].split("Label:")[-1].strip().splitlines()[0]


class HierarchicalClassifier:
    """Two-stage hierarchical classifier using parent and child models."""

    def __init__(self, parent_model: str, child_models: dict[str, str]):
        self.parent = pipeline("text-classification", model=resolve_model_name(parent_model))
        self.children = {
            label: pipeline("text-classification", model=resolve_model_name(model_name))
            for label, model_name in child_models.items()
        }

    def predict(self, text: str) -> dict[str, str]:
        parent_result = self.parent(text, truncation=True)[0]
        parent_label = parent_result["label"]
        child_model = self.children.get(parent_label)
        if child_model is None:
            return {"parent": parent_label, "child": "N/A"}
        child_result = child_model(text, truncation=True)[0]
        return {"parent": parent_label, "child": child_result["label"]}
