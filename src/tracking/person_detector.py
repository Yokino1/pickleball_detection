"""Optional YOLO person detection for contact-gated ball tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .types import PlayerDetection


class PersonDetector(ABC):
    """Interface for a detector that emits person boxes."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        ...


class YoloPersonDetector(PersonDetector):
    """Ultralytics person detector, normally backed by a COCO nano model."""

    def __init__(
        self,
        model_path: str | Path,
        person_class_id: int = 0,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        imgsz: int = 416,
        max_detections: int = 16,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.person_class_id = int(person_class_id)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.max_detections = int(max_detections)
        self.device = device
        self._model = None
        self.disabled = False
        self.last_error: Optional[str] = None

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        if self.disabled:
            return []
        try:
            if self._model is None:
                self._load_model()
            results = self._model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                classes=[self.person_class_id],
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.disabled = True
            self.last_error = str(exc)
            return []

        detections: list[PlayerDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                detections.append(
                    PlayerDetection(
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        foot_point=[(x1 + x2) / 2.0, y2],
                        source="person_yolo",
                    )
                )
        return detections

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Person model does not exist: {self.model_path}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))

