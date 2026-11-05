"""
VisionAssist 2.0 - Custom YOLO Detection Module

Provides a reusable wrapper around the project's custom-trained YOLO
model for running inference on individual frames. This module contains
no visualization, webcam, or execution logic — it is purely a
detection component to be used by other parts of the pipeline.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import CUSTOM_YOLO


class CustomYOLODetector:
    """Wrapper around the custom-trained YOLO model for object detection.

    This class handles model loading and validation, and exposes a
    simple `predict` method for running inference on a single frame.
    It does not perform any drawing, visualization, or frame mutation.

    Attributes:
        model (YOLO): The loaded Ultralytics YOLO model instance.
    """

    def __init__(self) -> None:
        """Initialize the detector by loading the custom YOLO model.

        Raises:
            FileNotFoundError: If the custom model weights file does
                not exist at the expected path.
        """
        # Verify the model file exists before attempting to load it
        if not CUSTOM_YOLO.exists():
            raise FileNotFoundError(
                f"Custom YOLO model not found at '{CUSTOM_YOLO}'. "
                "Please ensure 'best_v2.pt' is placed inside the 'models/' directory."
            )

        # Load the custom-trained YOLO model
        self.model: YOLO = YOLO(str(CUSTOM_YOLO))

    def predict(self, frame: np.ndarray) -> Results:
        """Run YOLO inference on a single frame.

        Args:
            frame (np.ndarray): An OpenCV BGR image (as a NumPy array)
                to run detection on. The frame is not modified.

        Returns:
            Results: The raw Ultralytics results object for the frame,
                containing detected boxes, classes, and confidences.
        """
        # Run inference; Ultralytics returns a list of Results (one per frame)
        results = self.model(
            frame,
            conf=0.45,
            verbose=False
        )
        return results[0]