"""
SPAZE - Room Extractor & Dimension Engine
-----------------------------------------
Enclosed region identification via contour hierarchy,
Douglas-Peucker polygon simplification, real-world metric scaling,
and transparent heuristic Quality Score calculation.
"""

from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from shapely.geometry import Polygon

from config import (
    ROOM_TYPES,
    ROOM_COLORS,
    WEIGHT_RECTANGULARITY,
    WEIGHT_SOLIDITY,
    WEIGHT_ASPECT_RATIO
)
from models import Room, ScaleInfo

class RoomExtractor:
    def __init__(
        self,
        min_room_area_px: int = 1500,
        max_room_area_px: int = 250000,
        epsilon_factor: float = 0.02
    ):
        self.min_room_area_px = min_room_area_px
        self.max_room_area_px = max_room_area_px
        self.epsilon_factor = epsilon_factor

    def compute_quality_score(
        self,
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Calculates Quality Score = 0.45 * R + 0.35 * S + 0.20 * A
        R: Rectangularity (contour area / bounding-box area)
        S: Solidity (contour area / convex-hull area)
        A: Aspect Ratio Score (rewards proportions between 0.5 and 2.0)
        """
        area = cv2.contourArea(contour)
        x, y, w, h = bbox
        bbox_area = max(float(w * h), 1.0)

        # 1. Rectangularity (R)
        r = min(area / bbox_area, 1.0)

        # 2. Solidity (S)
        hull = cv2.convexHull(contour)
        hull_area = max(cv2.contourArea(hull), 1.0)
        s = min(area / hull_area, 1.0)

        # 3. Aspect Ratio Score (A)
        ar = w / max(float(h), 1.0)
        if 0.5 <= ar <= 2.0:
            a = 1.0 - abs(ar - 1.0) * 0.4
        else:
            a = max(0.0, 1.0 - abs(ar - 1.0) * 0.6)

        # Total Weighted Quality Score
        quality_score = (
            WEIGHT_RECTANGULARITY * r +
            WEIGHT_SOLIDITY * s +
            WEIGHT_ASPECT_RATIO * a
        )

        return round(quality_score, 3), round(r, 3), round(s, 3), round(ar, 3)

    def extract_rooms(
        self,
        morphed_binary: np.ndarray,
        scale_info: ScaleInfo
    ) -> List[Room]:
        """
        Identifies enclosed room candidates by inverting wall boundaries,
        analyzing contour hierarchies, and computing metric attributes.
        """
        # Invert binary image so empty room cavities become foreground white
        room_cavities = cv2.bitwise_not(morphed_binary)

        # Find external and internal contours
        contours, hierarchy = cv2.findContours(
            room_cavities,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        rooms: List[Room] = []
        if not contours or hierarchy is None:
            return rooms

        h_img, w_img = morphed_binary.shape
        full_img_area = h_img * w_img

        room_idx = 1
        for i, cnt in enumerate(contours):
            area_px = cv2.contourArea(cnt)

            # Filter out tiny noise contours, text labels, or full outer image boundary
            if area_px < 5000 or area_px > (full_img_area * 0.70):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Must have minimum width and height to be a real architectural room (not text labels)
            if w < 70 or h < 70:
                continue

            # Ignore outer border margins
            if x <= 5 or y <= 5 or (x + w) >= (w_img - 5) or (y + h) >= (h_img - 5):
                continue

            # Douglas-Peucker Polygon Simplification
            epsilon = self.epsilon_factor * cv2.arcLength(cnt, True)
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            poly_pts = [(int(p[0][0]), int(p[0][1])) for p in approx_poly]

            # Quality Score calculation
            quality, r_val, s_val, a_val = self.compute_quality_score(cnt, (x, y, w, h))

            # Metric scale conversion (px -> metres)
            px_per_m = max(scale_info.px_per_metre, 1.0)
            width_m = w / px_per_m
            height_m = h / px_per_m
            area_m2 = area_px / (px_per_m ** 2)

            # Intelligent Room Classification Heuristics
            room_name, type_id = self._infer_room_type(area_m2, width_m, height_m, room_idx)

            rooms.append(Room(
                id=room_idx,
                name=room_name,
                type_id=type_id,
                bbox_px=(x, y, w, h),
                polygon_px=poly_pts,
                area_m2=round(area_m2, 2),
                width_m=round(width_m, 2),
                height_m=round(height_m, 2),
                rectangularity=r_val,
                solidity=s_val,
                aspect_ratio=a_val,
                quality_score=quality,
                color_hex=ROOM_COLORS.get(room_name, "#3B82F6")
            ))
            room_idx += 1

        return rooms

    def _infer_room_type(
        self,
        area_m2: float,
        width_m: float,
        height_m: float,
        idx: int
    ) -> Tuple[str, int]:
        """
        Classifies room category based on architectural area and aspect standards.
        """
        if area_m2 >= 20.0:
            return f"Living Room", 0
        elif 14.0 <= area_m2 < 20.0:
            return f"Master Bedroom", 1
        elif 9.0 <= area_m2 < 14.0:
            return f"Bedroom {idx}", 2
        elif 5.0 <= area_m2 < 9.0:
            return f"Kitchen", 3
        elif 2.0 <= area_m2 < 5.0:
            return f"Bathroom", 4
        elif area_m2 < 2.0:
            return f"Balcony", 5
        else:
            return f"Dining Area", 6
