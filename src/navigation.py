# src/navigation.py

from collections import deque
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class NavigationEngine:
    """
    Intelligent Anti-Jitter Navigation Decision Engine for VisionAssist 2.0.
    Fuses SegFormer walkable floor segmentation with YOLO obstacle locations
    using temporal sliding-window smoothing and directional hysteresis to produce
    reliable, human-like, chatter-free navigational decisions: FORWARD, LEFT, RIGHT, or STOP.
    """

    def __init__(self, history_size: int = 7):
        self.history_size = history_size
        self.decision_history: deque = deque(maxlen=history_size)
        self.stabilized_direction: str = "FORWARD"
        self.previous: str = "FORWARD"
        self.active_detour: Optional[str] = None  # Tracks active detour intent ('LEFT' or 'RIGHT')

    def reset(self) -> None:
        """Reset temporal history and state trackers."""
        self.decision_history.clear()
        self.stabilized_direction = "FORWARD"
        self.previous = "FORWARD"
        self.active_detour = None

    def decide_instantaneous(
        self,
        walkable_mask: Optional[np.ndarray],
        depth_map: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Evaluate single-frame navigation decision based on walkable floor and obstacles.
        """
        if walkable_mask is None or walkable_mask.size == 0:
            return "STOP"

        h, w = walkable_mask.shape[:2]

        # Ground ROI: bottom 45% of the frame directly in front of the user
        roi = walkable_mask[int(h * 0.55):, :]
        roi_h, roi_w = roi.shape[:2]

        third = roi_w // 3
        if third == 0 or roi_h == 0:
            return "STOP"

        sector_pixels = float(roi_h * third)
        if sector_pixels == 0:
            return "STOP"

        # Corridors
        left_sector = roi[:, :third]
        center_sector = roi[:, third:2 * third]
        right_sector = roi[:, 2 * third:]

        # Measure walkable floor ratio (0.0 - 1.0) in each corridor
        left_floor = cv2.countNonZero(left_sector) / sector_pixels
        center_floor = cv2.countNonZero(center_sector) / sector_pixels
        right_floor = cv2.countNonZero(right_sector) / sector_pixels

        # Check obstacle blockage in each corridor with soft boundary zones
        left_blocked = False
        center_blocked = False
        right_blocked = False

        if detected_objects:
            # Corridor split boundaries with slight deadband margins
            c_left_bound = int(w * 0.30)
            c_right_bound = int(w * 0.70)

            for obj in detected_objects:
                dist = obj.get("distance", 99.0)
                cx = obj.get("center_x", w // 2)

                # Center forward corridor: obstacle within 2.0m blocks forward movement
                if c_left_bound <= cx <= c_right_bound:
                    if dist <= 2.0:
                        center_blocked = True
                elif cx < c_left_bound:
                    if dist <= 1.6:
                        left_blocked = True
                else:
                    if dist <= 1.6:
                        right_blocked = True

        # Rule 1: Emergency Stop - if all sectors have virtually no walkable floor (< 4%) and center is blocked
        if center_floor < 0.04 and left_floor < 0.04 and right_floor < 0.04 and center_blocked:
            return "STOP"

        # Rule 2: Unblocked Center Corridor with adequate floor -> GO FORWARD
        if not center_blocked and (center_floor >= 0.05 or (center_floor >= left_floor and center_floor >= right_floor)):
            return "FORWARD"

        # Rule 3: Center corridor is blocked -> Steer Left or Right with Hysteresis
        if center_blocked or center_floor < 0.05:
            # If an active detour is already in progress, strongly favor keeping it unless blocked
            if self.active_detour == "LEFT" and not left_blocked and left_floor >= 0.05:
                return "LEFT"
            if self.active_detour == "RIGHT" and not right_blocked and right_floor >= 0.05:
                return "RIGHT"

            # Otherwise select best open corridor
            if not left_blocked and left_floor >= 0.06 and (left_floor >= right_floor or right_blocked):
                return "LEFT"
            elif not right_blocked and right_floor >= 0.06:
                return "RIGHT"
            elif not center_blocked:
                return "FORWARD"
            else:
                return "STOP"

        return "FORWARD"

    def decide(
        self,
        walkable_mask: Optional[np.ndarray],
        depth_map: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Produce a temporally smoothed, anti-flutter navigation decision.
        """
        raw_decision = self.decide_instantaneous(
            walkable_mask=walkable_mask,
            depth_map=depth_map,
            detected_objects=detected_objects
        )

        self.decision_history.append(raw_decision)

        # Immediate override for emergency STOP:
        # If STOP appears twice in history or raw is STOP with close obstacle, trigger STOP quickly
        if raw_decision == "STOP":
            stop_count = sum(1 for d in self.decision_history if d == "STOP")
            if stop_count >= 2 or len(self.decision_history) <= 2:
                self.stabilized_direction = "STOP"
                self.active_detour = None
                self.previous = "STOP"
                return "STOP"

        # Majority voting over sliding history buffer
        counts: Dict[str, int] = {}
        for d in self.decision_history:
            counts[d] = counts.get(d, 0) + 1

        # Current stabilized direction gets hysteresis bias (+1 vote) to prevent rapid alternating
        counts[self.stabilized_direction] = counts.get(self.stabilized_direction, 0) + 1

        # Determine winner
        best_direction = max(counts, key=lambda k: counts[k])

        # Require meaningful consensus to change stabilized direction
        # (needs at least half the window agreeing on the new direction)
        min_required = max(2, len(self.decision_history) // 2)
        raw_winner_count = sum(1 for d in self.decision_history if d == best_direction)

        if raw_winner_count >= min_required or len(self.decision_history) < 3:
            self.stabilized_direction = best_direction

        # Update active detour state
        if self.stabilized_direction in ("LEFT", "RIGHT"):
            self.active_detour = self.stabilized_direction
        elif self.stabilized_direction == "FORWARD":
            # Clear active detour once path is confirmed forward
            forward_count = sum(1 for d in self.decision_history if d == "FORWARD")
            if forward_count >= 4:
                self.active_detour = None

        self.previous = self.stabilized_direction
        return self.stabilized_direction

    def decide_from_depth(
        self,
        depth_map: np.ndarray,
        detected_objects: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Fallback depth-only decision method."""
        dummy_mask = np.full(depth_map.shape[:2], 255, dtype=np.uint8)
        return self.decide(dummy_mask, depth_map=depth_map, detected_objects=detected_objects)