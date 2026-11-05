"""
VisionAssist 2.0 - COCO YOLO Detection Module

Wrapper around the pretrained YOLOv8 COCO model for detecting
general obstacles such as people, chairs, tables, bicycles, and cars.
"""

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import COCO_YOLO


class CocoYOLODetector:
    """Pretrained COCO object detector."""

    def __init__(self) -> None:

        if not COCO_YOLO.exists():
            raise FileNotFoundError(
                f"COCO model not found at '{COCO_YOLO}'."
            )

        self.model = YOLO(str(COCO_YOLO))

    def predict(self, frame: np.ndarray) -> Results:
        """Run inference on a single frame."""

        results = self.model(
            frame,
            conf=0.45,
            verbose=False
        )
        return results[0]