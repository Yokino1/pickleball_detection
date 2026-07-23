"""
Optional player detection — pluggable, disabled by default.

The system must work fully without player detection.  When enabled,
players are drawn on the overlay but never affect ball tracking,
court projection, or event detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .types import PlayerDetection


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class PlayerDetector(ABC):
    """Interface for optional player detection."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        """Return 0–N player detections for *frame*."""
        ...


# ---------------------------------------------------------------------------
# Null detector (default)
# ---------------------------------------------------------------------------

class NullPlayerDetector(PlayerDetector):
    """Always returns an empty list — player detection disabled."""

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        return []


# ---------------------------------------------------------------------------
# YOLO-based player detector
# ---------------------------------------------------------------------------

class YoloPlayerDetector(PlayerDetector):
    """Player detector backed by a local YOLO model.

    Fails gracefully: if the model can't be loaded or inference fails,
    falls back to returning an empty list.
    """

    def __init__(
        self,
        model_path: str | Path,
        player_class_id: int = 0,
        conf_threshold: float = 0.3,
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        self._model_path = Path(model_path)
        self._player_class_id = player_class_id
        self._conf_threshold = conf_threshold
        self._imgsz = imgsz
        self._device = device
        self._model = None
        self._disabled = False

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        if self._disabled:
            return []
        try:
            return self._detect_impl(frame)
        except Exception:
            self._disabled = True
            return []

    def _detect_impl(self, frame: np.ndarray) -> list[PlayerDetection]:
        if self._model is None:
            self._load_model()

        import torch
        with torch.inference_mode():
            results = self._model(
                frame,
                imgsz=self._imgsz,
                conf=self._conf_threshold,
                verbose=False,
            )

        detections: list[PlayerDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != self._player_class_id:
                    continue
                conf = float(box.conf[0])
                if conf < self._conf_threshold:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                # Foot point = bottom-center of bbox
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                detections.append(PlayerDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    foot_point=[foot_x, foot_y],
                ))

        return detections

    def _load_model(self) -> None:
        if not self._model_path.exists():
            self._disabled = True
            return

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            self._disabled = True
            return

        self._model = YOLO(str(self._model_path))
        if self._device is not None:
            self._model.to(self._device)
