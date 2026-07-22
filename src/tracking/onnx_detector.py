"""Torch-free ONNX Runtime detector for edge deployment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .ball_detector import BallDetector
from .types import BallDetection


class OnnxBallDetector(BallDetector):
    """Run a one-class YOLO ONNX model without Ultralytics or PyTorch.

    Both end-to-end ``[x1, y1, x2, y2, confidence, class_id]`` output and
    conventional YOLO ``[cx, cy, width, height, class_scores...]`` output are
    supported. Exporting with embedded NMS is preferred for board deployment.
    """

    def __init__(
        self,
        model_path: str | Path,
        ball_class_id: int = 0,
        conf_threshold: float = 0.08,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        max_detections: int = 32,
        providers: Optional[list[str]] = None,
        exclude_region: Optional[list[float]] = None,
    ):
        self.model_path = Path(model_path)
        self.ball_class_id = int(ball_class_id)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.max_detections = int(max_detections)
        self.providers = providers
        self.exclude_region = exclude_region
        self._session = None
        self._input_name = ""
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def detect(
        self, frame: np.ndarray, roi: Optional[list[float]] = None
    ) -> list[BallDetection]:
        if self._disabled:
            return []
        try:
            return self._detect_impl(frame, roi)
        except Exception as exc:
            print(f"[OnnxBallDetector] disabled after failure: {exc}", file=sys.stderr)
            self._disabled = True
            return []

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("onnxruntime is required for ONNX inference") from exc
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        kwargs = {"sess_options": options}
        if self.providers:
            kwargs["providers"] = self.providers
        self._session = ort.InferenceSession(str(self.model_path), **kwargs)
        self._input_name = self._session.get_inputs()[0].name

    def _detect_impl(
        self, frame: np.ndarray, roi: Optional[list[float]]
    ) -> list[BallDetection]:
        if self._session is None:
            self._load()

        crop, offset_x, offset_y = self._crop(frame, roi)
        tensor, scale, pad_x, pad_y = self._preprocess(crop)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not outputs:
            return []
        boxes, scores, classes, end_to_end = self._decode(np.asarray(outputs[0]))
        if not len(boxes):
            return []

        if not end_to_end:
            boxes = self._xywh_to_xyxy(boxes)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale + offset_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale + offset_y
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.shape[0])

        keep = np.where((scores >= self.conf_threshold) & (classes == self.ball_class_id))[0]
        boxes, scores = boxes[keep], scores[keep]
        if not end_to_end:
            keep_nms = self._nms(boxes, scores)
            boxes, scores = boxes[keep_nms], scores[keep_nms]
        order = np.argsort(-scores)[: self.max_detections]

        detections = []
        for index in order:
            x1, y1, x2, y2 = boxes[index].tolist()
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            if self.exclude_region is not None:
                ex1, ey1, ex2, ey2 = self.exclude_region
                if ex1 <= center[0] <= ex2 and ey1 <= center[1] <= ey2:
                    continue
            detections.append(
                BallDetection(
                    bbox=[x1, y1, x2, y2],
                    center=center,
                    confidence=float(scores[index]),
                    source="onnxruntime",
                )
            )
        return detections

    def _crop(self, frame: np.ndarray, roi: Optional[list[float]]):
        if roi is None:
            return frame, 0.0, 0.0
        x1, y1, x2, y2 = map(int, roi)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return frame, 0.0, 0.0
        return frame[y1:y2, x1:x2], float(x1), float(y1)

    def _preprocess(self, image: np.ndarray):
        import cv2

        height, width = image.shape[:2]
        scale = min(self.imgsz / width, self.imgsz / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.imgsz - resized_width) / 2.0
        pad_y = (self.imgsz - resized_height) / 2.0
        left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
        right = self.imgsz - resized_width - left
        bottom = self.imgsz - resized_height - top
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return tensor, scale, float(left), float(top)

    def _decode(self, output: np.ndarray):
        prediction = np.squeeze(output)
        if prediction.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {output.shape}")
        if prediction.shape[1] == 6:
            boxes = prediction[:, :4].astype(np.float64, copy=False)
            scores = prediction[:, 4].astype(np.float64, copy=False)
            classes = prediction[:, 5].astype(np.int64, copy=False)
            return boxes, scores, classes, True
        if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 256:
            prediction = prediction.T

        if prediction.shape[1] == 6:
            boxes = prediction[:, :4].astype(np.float64, copy=False)
            scores = prediction[:, 4].astype(np.float64, copy=False)
            classes = prediction[:, 5].astype(np.int64, copy=False)
            return boxes, scores, classes, True
        if prediction.shape[1] < 5:
            raise ValueError(f"Unsupported YOLO prediction width: {prediction.shape[1]}")

        boxes = prediction[:, :4].astype(np.float64, copy=False)
        class_scores = prediction[:, 4:]
        classes = np.argmax(class_scores, axis=1).astype(np.int64)
        scores = class_scores[np.arange(len(class_scores)), classes].astype(np.float64)
        return boxes, scores, classes, False

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        result = boxes.copy()
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return result

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        if not len(boxes):
            return []
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(-scores)
        keep = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            intersection_width = np.maximum(0.0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest]))
            intersection_height = np.maximum(0.0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest]))
            intersection = intersection_width * intersection_height
            union = areas[current] + areas[rest] - intersection
            iou = intersection / np.maximum(union, 1e-9)
            order = rest[iou <= self.iou_threshold]
        return keep
