"""Object detection adapter implementations."""

from __future__ import annotations

from typing import Any

from multimodal_ai.adapters.base import DetectionAdapter


class YOLODetectionAdapter(DetectionAdapter):
    """Ultralytics YOLO adapter."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.name = "yolo"
        self._model_name = model_name
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)
        except Exception:  # noqa: BLE001
            self._model = None

    def health(self) -> dict[str, Any]:
        self._load()
        return {"ok": self._model is not None, "name": self.name, "model": self._model_name}

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        self._load()
        if self._model is None:
            return []

        results = self._model(image_path)
        detections: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = [float(v) for v in box.xyxy[0].tolist()]
                label = result.names.get(cls_id, str(cls_id))
                detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox": coords,
                    }
                )
        return detections


class GroundingDINODetectionAdapter(DetectionAdapter):
    """Optional grounding-dino placeholder adapter."""

    def __init__(self) -> None:
        self.name = "grounding_dino"

    def health(self) -> dict[str, Any]:
        return {"ok": False, "name": self.name, "reason": "optional_not_loaded"}

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        return []
