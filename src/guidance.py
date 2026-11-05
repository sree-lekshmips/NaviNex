# src/guidance.py

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ObstacleInfo:
    label: str
    distance: float
    danger: str          # 'DANGER', 'CAUTION', 'SAFE'
    center_x: int
    activity: str = ""   # e.g., 'Walking', 'Running', 'Sitting', ''


# Explicitly defined high-priority objects that should be spoken by their exact name
HIGH_PRIORITY_CLASSES = {
    # Persons / Pedestrians
    "person", "pedestrian", "man", "woman", "child", "people",
    # Vehicles
    "car", "bus", "truck", "motorcycle", "motorbike", "bicycle", "bike", "train", "boat", "airplane", "vehicle",
    # Stairs & steps (preserved strictly)
    "stairs", "stair", "staircase", "step", "steps",
    # Walls & barriers
    "wall",
    # Doors & portals
    "door", "doorway", "door-frame"
}


def get_spoken_label(raw_label: str) -> str:
    """
    Map raw detector/segmenter labels into spoken names according to priority rules:
    - High-priority objects (person, vehicle, stairs, wall, door) are spoken by name.
    - Small / generic / indoor objects (chair, table, bottle, cup, laptop, etc.) are referred to strictly as 'obstacle'.
    """
    label = raw_label.lower().strip()

    if "stair" in label or "step" in label:
        return "stairs"
    if "door" in label:
        return "door"
    if label == "wall":
        return "wall"
    if label in ("person", "pedestrian", "man", "woman", "child", "people"):
        return "person"
    if label in ("car", "bus", "truck", "motorcycle", "motorbike", "bicycle", "bike", "vehicle", "train", "boat", "airplane"):
        return label
    if label in HIGH_PRIORITY_CLASSES:
        return label

    # All small, generic, or indoor items are spoken as 'obstacle'
    return "obstacle"


class GuidanceEngine:
    """
    Intelligent Obstacle Reasoning & Natural Guidance Engine for VisionAssist 2.0.
    Combines object detections, metric distances, safe directions, scene, and activity
    to generate natural, contextual, flutter-free spoken instructions with state hysteresis.
    """

    def __init__(self, frame_width: int = 1280, min_clear_frames: int = 12, min_obstacle_frames: int = 3):
        self.frame_width = frame_width
        self.min_clear_frames = min_clear_frames
        self.min_obstacle_frames = min_obstacle_frames

        # State trackers: 'INIT', 'CLEAR', 'OBSTACLE', 'STOP'
        self.state: str = "INIT"
        self.last_spoken_instruction: str = ""
        self.last_spoken_time: float = 0.0

        # Confirmation counters to prevent flapping
        self.consecutive_clear_frames: int = 0
        self.consecutive_obstacle_frames: int = 0
        self.current_obstacle_signature: str = ""

        # Detour direction lock
        self.locked_turn: Optional[str] = None

    def reset(self) -> None:
        """Reset guidance engine counters and state."""
        self.state = "INIT"
        self.last_spoken_instruction = ""
        self.consecutive_clear_frames = 0
        self.consecutive_obstacle_frames = 0
        self.current_obstacle_signature = ""
        self.locked_turn = None

    def _should_speak(self, spoken_text: str, is_urgent: bool = False) -> bool:
        """
        Check if speech should be emitted based on sentence change and minimum hold.
        """
        if not spoken_text:
            return False
        if spoken_text == self.last_spoken_instruction:
            return False
        return True

    def generate_guidance(
        self,
        objects: List[ObstacleInfo],
        direction: str = "FORWARD",
        corridor_clearance: float = 3.0,
        scene: str = "",
        human_activity: str = ""
    ) -> Tuple[str, Optional[str]]:
        """
        Generate both HUD display text and state-aware, stabilized spoken guidance.

        Key Improvements:
        - Prevents jittery alternating sentences (e.g. 'Move left' <-> 'Path clear' <-> 'Move right').
        - Requires a continuous clear path for min_clear_frames before announcing 'Path is clear'.
        - Urgent emergency stops and stairs announce immediately.
        - Moderate obstacle alerts require consecutive frame confirmation.
        """
        now = time.time()

        # -------------------------------------------------------------
        # 1. Emergency Safety Priority (Immediate STOP)
        # -------------------------------------------------------------
        if direction == "STOP":
            self.consecutive_clear_frames = 0
            self.consecutive_obstacle_frames += 1
            self.locked_turn = None

            if objects:
                nearest = min(objects, key=lambda x: x.distance)
                spoken_name = get_spoken_label(nearest.label)
                display_text = f"Warning: Path blocked by {spoken_name}. Please stop."
                spoken_text = f"Warning. Path blocked by {spoken_name}. Please stop and be careful."
            else:
                display_text = "Warning: No walkable path ahead. Please stop."
                spoken_text = "Warning. No walkable path ahead. Please stop and turn around."

            self.state = "STOP"
            if self._should_speak(spoken_text, is_urgent=True):
                self.last_spoken_instruction = spoken_text
                self.last_spoken_time = now
                return display_text, spoken_text
            return display_text, None

        # -------------------------------------------------------------
        # 2. Critical Safety Hazards (Stairs / Steps)
        # -------------------------------------------------------------
        for obj in objects:
            if "stair" in obj.label.lower() and obj.distance <= 3.0:
                self.consecutive_clear_frames = 0
                self.consecutive_obstacle_frames += 1
                display_text = "Caution: Stairs detected ahead."
                spoken_text = "Caution. Stairs detected ahead. Proceed with care."
                self.state = "OBSTACLE"
                if self._should_speak(spoken_text, is_urgent=True):
                    self.last_spoken_instruction = spoken_text
                    self.last_spoken_time = now
                    return display_text, spoken_text
                return display_text, None

        # -------------------------------------------------------------
        # 3. Immediate Danger Obstacles (< 1.2 meters)
        # -------------------------------------------------------------
        danger_objs = [obj for obj in objects if obj.distance < 1.2]
        if danger_objs:
            self.consecutive_clear_frames = 0
            self.consecutive_obstacle_frames += 1

            nearest = min(danger_objs, key=lambda x: x.distance)
            spoken_name = get_spoken_label(nearest.label)

            if spoken_name == "door":
                display_text = "Door reached ahead. Approach carefully."
                spoken_text = "Door reached ahead. Approach carefully."
            elif spoken_name == "wall":
                turn_phrase = "Move left." if direction == "LEFT" else "Move right." if direction == "RIGHT" else "Please stop."
                display_text = f"Wall detected ahead. {turn_phrase}"
                spoken_text = f"Wall detected ahead. {turn_phrase}"
            else:
                act_prefix = f"{nearest.activity.lower()} " if nearest.activity and spoken_name == "person" else ""
                turn_phrase = (
                    "Move left." if direction == "LEFT"
                    else "Move right." if direction == "RIGHT"
                    else "Be careful."
                )
                display_text = f"{act_prefix.capitalize()}{spoken_name.capitalize()} ahead. {turn_phrase}"
                spoken_text = f"{act_prefix.capitalize()}{spoken_name.capitalize()} detected ahead. {turn_phrase}"

            self.state = "OBSTACLE"
            # Require minimal confirmation or immediate if very close (< 0.9m)
            if nearest.distance < 0.9 or self.consecutive_obstacle_frames >= 2:
                if self._should_speak(spoken_text, is_urgent=(nearest.distance < 0.9)):
                    self.last_spoken_instruction = spoken_text
                    self.last_spoken_time = now
                    return display_text, spoken_text
            return display_text, None

        # -------------------------------------------------------------
        # 4. Turn Maneuvers around Obstacles (Direction is LEFT or RIGHT)
        # -------------------------------------------------------------
        if direction in ("LEFT", "RIGHT"):
            self.consecutive_clear_frames = 0
            self.consecutive_obstacle_frames += 1

            if direction == "LEFT":
                blocking_objs = [
                    obj for obj in objects
                    if obj.distance <= 1.8 and obj.center_x >= int(self.frame_width * 0.30) and obj.label.lower() != "wall"
                ]
                if blocking_objs:
                    nearest = min(blocking_objs, key=lambda x: x.distance)
                    spoken_name = get_spoken_label(nearest.label)
                    display_text = f"{spoken_name.capitalize()} on the right. Move left."
                    spoken_text = f"{spoken_name.capitalize()} on the right. Move left."
                else:
                    display_text = "Clear path on the left. Move left."
                    spoken_text = "Clear path on the left. Move left."
            else:  # RIGHT
                blocking_objs = [
                    obj for obj in objects
                    if obj.distance <= 1.8 and obj.center_x <= int(self.frame_width * 0.70) and obj.label.lower() != "wall"
                ]
                if blocking_objs:
                    nearest = min(blocking_objs, key=lambda x: x.distance)
                    spoken_name = get_spoken_label(nearest.label)
                    display_text = f"{spoken_name.capitalize()} on the left. Move right."
                    spoken_text = f"{spoken_name.capitalize()} on the left. Move right."
                else:
                    display_text = "Clear path on the right. Move right."
                    spoken_text = "Clear path on the right. Move right."

            self.state = "OBSTACLE"
            # Require at least min_obstacle_frames (3 frames) to prevent 1-frame transient turn shouts
            if self.consecutive_obstacle_frames >= self.min_obstacle_frames:
                if self._should_speak(spoken_text):
                    self.last_spoken_instruction = spoken_text
                    self.last_spoken_time = now
                    return display_text, spoken_text
            return display_text, None

        # -------------------------------------------------------------
        # 5. Forward Corridor Obstacle (1.2m - 2.0m)
        # -------------------------------------------------------------
        center_objs = [
            obj for obj in objects
            if obj.distance <= 2.0 and int(self.frame_width * 0.25) <= obj.center_x <= int(self.frame_width * 0.75)
        ]
        if center_objs:
            self.consecutive_clear_frames = 0
            self.consecutive_obstacle_frames += 1

            nearest = min(center_objs, key=lambda x: x.distance)
            spoken_name = get_spoken_label(nearest.label)
            act_prefix = f"{nearest.activity.lower()} " if nearest.activity and spoken_name == "person" else ""
            display_text = f"{act_prefix.capitalize()}{spoken_name.capitalize()} ahead. Be careful."
            spoken_text = f"{act_prefix.capitalize()}{spoken_name.capitalize()} detected ahead. Be careful."

            self.state = "OBSTACLE"
            if self.consecutive_obstacle_frames >= self.min_obstacle_frames:
                if self._should_speak(spoken_text):
                    self.last_spoken_instruction = spoken_text
                    self.last_spoken_time = now
                    return display_text, spoken_text
            return display_text, None

        # -------------------------------------------------------------
        # 6. Forward Path is Clear (Transition with Sustained Confirmation)
        # -------------------------------------------------------------
        self.consecutive_obstacle_frames = 0
        self.consecutive_clear_frames += 1

        display_text = "Path is clear ahead. Continue forward."
        spoken_text = "Path is clear, continue forward."

        # Only speak 'Path is clear' if transitioning from non-CLEAR state AND
        # the clear state has been sustained for at least min_clear_frames (~12-15 frames)
        if self.state != "CLEAR":
            if self.consecutive_clear_frames >= self.min_clear_frames:
                self.state = "CLEAR"
                if self._should_speak(spoken_text):
                    self.last_spoken_instruction = spoken_text
                    self.last_spoken_time = now
                    return display_text, spoken_text
            # While confirming clear status, show on HUD but do not speak yet
            return display_text, None

        # Path was already stably CLEAR -> keep silent
        return display_text, None

    def generate_instruction(
        self,
        objects: List[ObstacleInfo],
        direction: str = "FORWARD",
        corridor_clearance: float = 3.0,
        scene: str = "",
        human_activity: str = ""
    ) -> str:
        """Backward-compatible helper returning the HUD display string."""
        display_text, _ = self.generate_guidance(
            objects=objects,
            direction=direction,
            corridor_clearance=corridor_clearance,
            scene=scene,
            human_activity=human_activity
        )
        return display_text
