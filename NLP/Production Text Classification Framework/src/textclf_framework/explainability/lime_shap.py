"""LIME and SHAP explainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    from lime.lime_text import LimeTextExplainer
except Exception:  # pragma: no cover
    LimeTextExplainer = None


@dataclass(slots=True)
class ExplanationResult:
    method: str
    label: int
    confidence: float
    features: list[tuple[str, float]]


class TextExplainer:
    """Unified explainability wrapper around LIME and SHAP."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self._lime = LimeTextExplainer(class_names=class_names) if LimeTextExplainer is not None else None

    def explain_with_lime(
        self,
        text: str,
        predict_proba_fn: Callable[[list[str]], np.ndarray],
        num_features: int = 10,
    ) -> ExplanationResult:
        """Explain single prediction with LIME."""
        if self._lime is None:
            raise RuntimeError("LIME is unavailable; install lime dependency.")

        probs = predict_proba_fn([text])[0]
        pred = int(np.argmax(probs))
        explanation = self._lime.explain_instance(
            text,
            classifier_fn=predict_proba_fn,
            num_features=num_features,
            labels=[pred],
        )
        features = explanation.as_list(label=pred)
        return ExplanationResult(
            method="lime",
            label=pred,
            confidence=float(probs[pred]),
            features=[(str(token), float(score)) for token, score in features],
        )

    def explain_with_shap(
        self,
        texts: list[str],
        predict_proba_fn: Callable[[list[str]], np.ndarray],
    ) -> shap.Explanation:
        """Explain batch predictions with SHAP text masker."""
        if shap is None:
            raise RuntimeError("SHAP is unavailable; install shap dependency.")

        masker = shap.maskers.Text()
        explainer = shap.Explainer(predict_proba_fn, masker=masker, output_names=self.class_names)
        return explainer(texts)
