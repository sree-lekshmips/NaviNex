# src/feature_logger.py
import csv
import os
import cv2
import numpy as np

MAX_DIST = 8.0
FIELDS = ["video", "frame",
          "floor_L", "floor_C", "floor_R",
          "dist_L", "dist_C", "dist_R",
          "clearance", "n_obstacles",
          "rule_raw", "rule_final"]


class FeatureLogger:
    """Writes one row of navigation features per frame to a CSV file."""

    def __init__(self, path: str = "outputs/features.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if is_new:
            self.w.writerow(FIELDS)

    @staticmethod
    def _floor_shares(walkable_mask):
        if walkable_mask is None or walkable_mask.size == 0:
            return 0.0, 0.0, 0.0
        h = walkable_mask.shape[0]
        roi = walkable_mask[int(h * 0.55):, :]
        third = roi.shape[1] // 3
        if third == 0 or roi.shape[0] == 0:
            return 0.0, 0.0, 0.0
        area = float(roi.shape[0] * third)
        left = cv2.countNonZero(roi[:, :third]) / area
        centre = cv2.countNonZero(roi[:, third:2 * third]) / area
        right = cv2.countNonZero(roi[:, 2 * third:]) / area
        return left, centre, right

    @staticmethod
    def _zone_distances(objects, frame_w):
        d = {"L": MAX_DIST, "C": MAX_DIST, "R": MAX_DIST}
        for o in objects:
            dist = min(float(o.get("distance", MAX_DIST)), MAX_DIST)
            cx = o.get("center_x", frame_w // 2)
            if cx < frame_w * 0.30:
                zone = "L"
            elif cx <= frame_w * 0.70:
                zone = "C"
            else:
                zone = "R"
            d[zone] = min(d[zone], dist)
        return d["L"], d["C"], d["R"]

    def log(self, video, frame_idx, walkable_mask, objects, frame_w,
            clearance, rule_raw, rule_final):
        fl, fc, fr = self._floor_shares(walkable_mask)
        dl, dc, dr = self._zone_distances(objects, frame_w)
        try:
            clr = min(float(clearance), MAX_DIST)
        except (TypeError, ValueError):
            clr = MAX_DIST
        self.w.writerow([video, frame_idx,
                         round(fl, 4), round(fc, 4), round(fr, 4),
                         round(dl, 2), round(dc, 2), round(dr, 2),
                         round(clr, 2), len(objects), rule_raw, rule_final])
        self.f.flush()

    def close(self):
        self.f.close()