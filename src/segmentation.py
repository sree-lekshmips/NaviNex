# src/segmentation.py

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from config import SEGFORMER_MODEL_DIR


class SegFormerSegmenter:
    """
    SegFormer-B0 inference for VisionAssist 2.0.
    Produces a walkable floor mask, safe navigation directions, and visual overlay.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        target_path = Path(model_path) if model_path else SEGFORMER_MODEL_DIR

        if not target_path.exists():
            raise FileNotFoundError(
                f"SegFormer model not found at '{target_path}'."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load HuggingFace processor and segmentation model
        self.processor = SegformerImageProcessor.from_pretrained(str(target_path))
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(target_path))
        self.model.to(self.device)
        self.model.eval()

        # Dynamically determine class IDs from the model config
        self.FLOOR_ID = 0
        self.WALL_ID = 1
        self.DOOR_ID = 2
        self.STAIRS_ID = 3
        self.FURNITURE_ID = 4
        self.OBSTACLE_ID = 5

        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            for str_id, label_name in self.model.config.id2label.items():
                name = label_name.lower().strip()
                cid = int(str_id)
                if name == "floor":
                    self.FLOOR_ID = cid
                elif name == "wall":
                    self.WALL_ID = cid
                elif name == "door":
                    self.DOOR_ID = cid
                elif "stair" in name:
                    self.STAIRS_ID = cid
                elif name == "furniture":
                    self.FURNITURE_ID = cid
                elif name == "obstacle":
                    self.OBSTACLE_ID = cid

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on a BGR frame and return the 2D class segmentation map.
        Uses native 384x384 square resolution for maximum floor detection accuracy.
        
        Args:
            frame: Input BGR image (OpenCV format).
            
        Returns:
            2D numpy array (H, W) of class indices matching the frame's original size.
        """
        h, w = frame.shape[:2]

        # Use square 384x384 matching model's native training config
        square_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(
            images=rgb,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        small_seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Nearest-neighbor resize to native frame dimensions
        seg = cv2.resize(small_seg, (w, h), interpolation=cv2.INTER_NEAREST)
        return seg

    def get_walkable_mask(
        self,
        seg_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert segmentation map into a binary walkable mask (255 = walkable, 0 = obstacle/background).
        Includes an adaptive ground plane fallback for videos with unusual floor textures.
        
        Args:
            seg_map: 2D segmentation map.
            depth_map: Optional depth map for ground slope verification.
            
        Returns:
            Binary mask (uint8) with same dimensions.
        """
        if seg_map is None:
            return np.zeros((100, 100), dtype=np.uint8)

        h, w = seg_map.shape[:2]
        raw_floor = (seg_map == self.FLOOR_ID).astype(np.uint8) * 255

        # Check ground ROI coverage (bottom 45%)
        roi = raw_floor[int(h * 0.55):, :]
        floor_ratio = cv2.countNonZero(roi) / float(roi.size)

        # If SegFormer successfully identified walkable floor (>= 6% coverage), use it
        if floor_ratio >= 0.06:
            return raw_floor

        # Fallback for videos with unfamiliar floor textures:
        # Construct adaptive ground corridor in the lower ground region
        fallback_mask = raw_floor.copy()
        ground_y1 = int(h * 0.55)
        corridor_x1 = int(w * 0.15)
        corridor_x2 = int(w * 0.85)

        # Enable lower ground corridor
        fallback_mask[ground_y1:, corridor_x1:corridor_x2] = 255

        # Remove semantic obstacles from the corridor
        wall_mask = (seg_map == self.WALL_ID)
        furniture_mask = (seg_map == self.FURNITURE_ID)
        obstacle_mask = (seg_map == self.OBSTACLE_ID)

        blocked = wall_mask | furniture_mask | obstacle_mask
        fallback_mask[blocked] = 0

        # Remove very close depth obstacles
        if depth_map is not None and depth_map.shape[:2] == (h, w):
            fallback_mask[depth_map > 195] = 0

        return fallback_mask

    def get_wall_mask(self, seg_map: np.ndarray) -> np.ndarray:
        """Extract a binary mask of detected wall pixels (255 = wall, 0 = other)."""
        if seg_map is None:
            return np.zeros((100, 100), dtype=np.uint8)
        return (seg_map == self.WALL_ID).astype(np.uint8) * 255

    def extract_wall_obstacles(
        self,
        seg_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        min_area_ratio: float = 0.06
    ) -> list:
        """
        Extract genuine blocking wall obstacles from the segmentation map.
        Ignores side walls running along the left or right of a hallway/room.
        Only flags walls positioned directly ahead in the central walking corridor that block forward motion.
        
        Args:
            seg_map: 2D segmentation class index map.
            depth_map: Optional relative depth map from MiDaS.
            min_area_ratio: Minimum fraction of frame area to consider a valid wall segment.
            
        Returns:
            List of dicts with: label ('wall'), distance, danger, center_x, box (x1, y1, x2, y2).
        """
        if seg_map is None:
            return []

        h, w = seg_map.shape[:2]
        wall_mask = (seg_map == self.WALL_ID).astype(np.uint8) * 255
        min_pixels = int(h * w * min_area_ratio)

        # Morphological close to bridge small cracks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        wall_cleaned = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(wall_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wall_obstacles = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_pixels:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            x2 = x + bw
            y2 = y + bh
            cx = x + bw // 2

            # Filter 1: Wall must be positioned directly in the central forward walking corridor (32% to 68% width)
            # Walls located on the left flank (< 32%) or right flank (> 68%) are side hallway walls, NOT blocking obstacles
            if cx < int(w * 0.32) or cx > int(w * 0.68):
                continue

            # Filter 2: Wall must extend into the lower half of the frame (ground walking zone)
            if y2 < int(h * 0.52) or bh < int(h * 0.25):
                continue

            # Estimate distance by sampling depth in the central base zone of the wall
            dist_m = 2.5
            if depth_map is not None and depth_map.shape[:2] == (h, w):
                sample_y1 = max(y, y + int(bh * 0.50))
                sample_y2 = min(h, y2)
                sample_x1 = max(0, x + int(bw * 0.20))
                sample_x2 = min(w, x2 - int(bw * 0.20))

                roi_depth = depth_map[sample_y1:sample_y2, sample_x1:sample_x2]
                if roi_depth.size > 0:
                    med_depth = float(np.percentile(roi_depth, 75))
                    clamped_depth = max(15.0, min(255.0, med_depth))
                    dist_m = round(max(0.5, min(6.0, (160.0 / (clamped_depth + 10.0)) + 0.30)), 1)

            # Filter 3: Only consider forward blocking walls that are close (<= 1.5m)
            if dist_m > 1.5:
                continue

            danger = "DANGER" if dist_m < 1.1 else "CAUTION"

            wall_obstacles.append({
                "label": "wall",
                "distance": dist_m,
                "danger": danger,
                "center_x": cx,
                "box": (x, y, x2, y2)
            })

        return wall_obstacles

    def overlay(
        self,
        frame: np.ndarray,
        seg_map: Optional[np.ndarray],
        walkable_mask: Optional[np.ndarray] = None,
        alpha: float = 0.35
    ) -> np.ndarray:
        """
        Overlay walkable floor in bright translucent green onto the original frame.
        
        Args:
            frame: Original BGR frame.
            seg_map: 2D segmentation map.
            walkable_mask: Optional precomputed binary walkable mask.
            alpha: Blending weight for overlay (0.0 to 1.0).
            
        Returns:
            Blended BGR frame.
        """
        if frame is None:
            return frame

        h, w = frame.shape[:2]

        if walkable_mask is not None and walkable_mask.shape[:2] == (h, w):
            floor_mask = (walkable_mask > 0)
        elif seg_map is not None:
            if seg_map.shape[:2] != (h, w):
                seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
            floor_mask = (seg_map == self.FLOOR_ID)
        else:
            return frame

        if not np.any(floor_mask):
            return frame

        output = frame.copy()
        # Vibrant green overlay color (BGR: 20, 220, 60)
        green_tint = np.zeros_like(frame, dtype=np.uint8)
        green_tint[:] = (20, 220, 60)

        # Blend only onto the walkable mask pixels
        output[floor_mask] = cv2.addWeighted(frame[floor_mask], 1.0 - alpha, green_tint[floor_mask], alpha, 0)
        return output
