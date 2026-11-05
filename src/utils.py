"""
VisionAssist 2.0 - Utility Functions

Reusable helper functions used across the application.
"""

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize an OpenCV frame to the desired dimensions.

    Args:
        frame: Input BGR frame.
        width: Target width.
        height: Target height.

    Returns:
        Resized frame.
    """
    return cv2.resize(frame, (width, height))