"""
Ball detector interface and built-in implementations.

The detector is intentionally narrow: it only finds balls.  Player / court /
pose detection belong in separate modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .types import BallDetection


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class BallDetector(ABC):
    """Stable interface that every ball detector must satisfy.

    ``detect()`` receives a full-frame image (BGR numpy array) and an
    optional ROI ``[x1, y1, x2, y2]`` in **full-frame** coordinates.
    """

    @abstractmethod
    def detect(
        self, frame, roi: Optional[list[float]] = None
    ) -> list[BallDetection]:
        """Return 0–N ball detections for *frame*.

        Args:
            frame: BGR image as a numpy array (H, W, 3).
            roi: Optional search region [x1, y1, x2, y2] in full-frame px.
                 Ignored by current implementation (full-frame inference).

        Returns:
            List of ``BallDetection`` objects (may be empty).
        """
        ...


# ---------------------------------------------------------------------------
# Null detector — lets the pipeline run without a model
# ---------------------------------------------------------------------------

class NullBallDetector(BallDetector):
    """A detector that always returns an empty list.

    Use this to validate the pipeline, JSONL output, and overlay rendering
    before a real ball model is available.
    """

    def detect(
        self, frame, roi: Optional[list[float]] = None
    ) -> list[BallDetection]:
        return []


# ---------------------------------------------------------------------------
# YOLO detector — wraps an ultralytics YOLO model for ball detection
# ---------------------------------------------------------------------------

class YoloBallDetector(BallDetector):
    """Ball detector backed by a local YOLO model.

    The model is loaded lazily on the first ``detect()`` call so that
    constructing the detector does not trigger heavy imports.

    Parameters
    ----------
    model_path : str or Path
        Path to a ``.pt`` or ``.onnx`` YOLO model.
    ball_class_id : int
        Class index that corresponds to the ball.  For the legacy
        ``player_best.pt`` model this is 1 (Pickleball).
    conf_threshold : float
        Minimum confidence for a detection to be returned.
    imgsz : int
        Inference image size (longest side).  Smaller = faster.
    device : str
        ``"cpu"``, ``"cuda:0"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_path: str | Path,
        ball_class_id: int = 1,
        conf_threshold: float = 0.15,
        imgsz: int = 640,
        device: Optional[str] = None,
        exclude_region: Optional[list[float]] = None,
    ):
        self._model_path = Path(model_path)
        self._ball_class_id = ball_class_id
        self._conf_threshold = conf_threshold
        self._imgsz = imgsz
        self._device = device
        self._model = None  # lazy
        self._disabled = False
        # Optional exclusion zone [x1, y1, x2, y2] in full-frame px.
        # Detections whose center falls inside are discarded.
        self._exclude_region = exclude_region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def disabled(self) -> bool:
        return self._disabled

    def detect(
        self, frame: np.ndarray, roi: Optional[list[float]] = None
    ) -> list[BallDetection]:
        """Run YOLO inference and return ball detections in full-frame coords."""
        if self._disabled:
            return []
        try:
            return self._detect_impl(frame, roi)
        except Exception as exc:
            import sys
            print(f"[YoloBallDetector] disabled after failure: {exc}", file=sys.stderr)
            self._disabled = True
            return []

    def _detect_impl(
        self, frame: np.ndarray, roi: Optional[list[float]]
    ) -> list[BallDetection]:
        if self._model is None:
            self._load_model()

        # --- vertical ROI crop (keep full width, restrict y only) ----
        offset_x, offset_y = 0.0, 0.0
        input_frame = frame

        if roi is not None:
            _, y1, _, y2 = [max(0, int(v)) for v in roi]
            h, w = frame.shape[:2]
            # Add 10% vertical padding
            pad = int((y2 - y1) * 0.10)
            y1 = max(0, y1 - pad)
            y2 = min(h, y2 + pad)

            if y2 > y1:
                input_frame = frame[y1:y2, :]  # full width, restricted height
                offset_y = float(y1)

        # --- inference ------------------------------------------------
        import torch

        with torch.inference_mode():
            results = self._model(
                input_frame,
                imgsz=self._imgsz,
                conf=self._conf_threshold,
                verbose=False,
            )

        # --- collect ball detections ----------------------------------
        detections: list[BallDetection] = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != self._ball_class_id:
                    continue

                conf = float(box.conf[0])
                if conf < self._conf_threshold:
                    continue

                # xyxy → shift y back to full-frame
                x1b, y1b, x2b, y2b = [float(v) for v in box.xyxy[0]]
                y1f = y1b + offset_y
                y2f = y2b + offset_y

                cx = (x1b + x2b) / 2.0
                cy = (y1f + y2f) / 2.0

                # --- exclusion zone check ---
                if self._exclude_region is not None:
                    ex1, ey1, ex2, ey2 = self._exclude_region
                    if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                        continue  # skip detection in excluded area

                detections.append(BallDetection(
                    bbox=[x1b, y1f, x2b, y2f],
                    center=[cx, cy],
                    confidence=conf,
                    source="yolo",
                ))

        return detections

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the YOLO model (imports ultralytics only when needed)."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Ball model not found: {self._model_path}\n"
                "Set models.ball_model to a valid .pt file or 'null' to use "
                "NullBallDetector."
            )

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            missing = exc.name or "ultralytics"
            raise RuntimeError(
                f"Cannot load ball model because '{missing}' is not installed. "
                "Install ultralytics and torch, or set models.ball_model to 'null'."
            ) from exc

        self._model = YOLO(str(self._model_path))
        if self._device is not None:
            self._model.to(self._device)
