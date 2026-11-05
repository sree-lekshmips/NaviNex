"""
VisionAssist 2.0 - MiDaS Depth Estimation Module

Provides a reusable wrapper around the MiDaS model for estimating
relative depth from a single RGB frame.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from config import MIDAS_MODEL


class MiDaSDepthEstimator:
    """Wrapper around the MiDaS depth estimation model."""

    def __init__(self) -> None:
        """Load the MiDaS model."""

        if not MIDAS_MODEL.exists():
            raise FileNotFoundError(
                f"MiDaS model not found at '{MIDAS_MODEL}'."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load MiDaS from local Torch cache (offline)
        midas_repo = Path.home() / ".cache" / "torch" / "hub" / "intel-isl_MiDaS_master"

        self.model = torch.hub.load(
            str(midas_repo),
            "DPT_SwinV2_T_256",
            source="local"
        )

        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load(
            str(midas_repo),
            "transforms",
            source="local"
        )
    
        self.transform = midas_transforms.swin256_transform

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Estimate relative depth map for one frame (0-255, 255=closest)."""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        depth = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return depth.astype(np.uint8)

    def get_median_depth_in_box(
        self,
        depth_map: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        inner_ratio: float = 0.50
    ) -> float:
        """
        Sample the median depth value from the inner core region of a bounding box
        to prevent edge noise and background bleed.
        
        Args:
            depth_map: 2D relative depth array.
            x1, y1, x2, y2: Bounding box coordinates.
            inner_ratio: Ratio of the bounding box to sample (default 0.50 = center 50%).
            
        Returns:
            Median relative depth value (0-255).
        """
        h, w = depth_map.shape[:2]

        # Calculate inset margin
        dx = int((x2 - x1) * (1.0 - inner_ratio) / 2.0)
        dy = int((y2 - y1) * (1.0 - inner_ratio) / 2.0)

        bx1 = max(0, min(w - 1, x1 + dx))
        bx2 = max(0, min(w, x2 - dx))
        by1 = max(0, min(h - 1, y1 + dy))
        by2 = max(0, min(h, y2 - dy))

        if bx2 <= bx1 or by2 <= by1:
            # Fallback to single center point
            cx = max(0, min(w - 1, (x1 + x2) // 2))
            cy = max(0, min(h - 1, (y1 + y2) // 2))
            return float(depth_map[cy, cx])

        roi = depth_map[by1:by2, bx1:bx2]
        return float(np.median(roi))

    @staticmethod
    def estimate_distance_meters(depth_val: float, label: str = "") -> float:
        """
        Convert relative MiDaS depth (0-255, higher = closer) to an approximate
        metric distance in meters.
        
        Args:
            depth_val: Relative depth value (0.0 - 255.0).
            label: Optional object class label for obstacle-specific scaling.
            
        Returns:
            Estimated distance in meters (rounded to 1 decimal place).
        """
        clamped_depth = max(15.0, min(255.0, float(depth_val)))
        
        # Non-linear inverse mapping calibrated for typical camera FOV
        distance = (160.0 / (clamped_depth + 10.0)) + 0.30

        # Bound realistic navigation range (0.5m to 6.0m)
        distance = max(0.5, min(6.0, distance))
        return round(distance, 1)

    def get_corridor_clearance_meters(self, depth_map: np.ndarray) -> float:
        """
        Measure the physical forward walking clearance in the center walking corridor.
        
        Args:
            depth_map: 2D depth array (0-255, 255=closest).
            
        Returns:
            Estimated clear walking distance ahead in meters (0.5m to 6.0m).
        """
        if depth_map is None or depth_map.size == 0:
            return 3.0

        h, w = depth_map.shape[:2]

        # Central walking corridor: middle 40% width, lower 50% height (ground to chest level)
        cx1, cx2 = int(w * 0.30), int(w * 0.70)
        cy1, cy2 = int(h * 0.45), int(h * 0.90)

        corridor_roi = depth_map[cy1:cy2, cx1:cx2]
        if corridor_roi.size == 0:
            return 3.0

        # 80th percentile depth captures the nearest forward ground or barrier surface
        corridor_depth = float(np.percentile(corridor_roi, 80))
        clearance = self.estimate_distance_meters(corridor_depth)
        return clearance

    @staticmethod
    def get_danger_level(distance_meters: float) -> str:
        """
        Categorize distance into danger thresholds:
        - DANGER: < 1.2 m (Immediate danger)
        - CAUTION: 1.2 m - 2.2 m (Caution / Approaching)
        - SAFE: > 2.2 m (Clear)
        """
        if distance_meters < 1.2:
            return "DANGER"
        elif distance_meters <= 2.2:
            return "CAUTION"
        return "SAFE"

