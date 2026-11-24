"""
SPAZE - Wall Detector Module
----------------------------
Extracts structural wall segments via Probabilistic Hough Line Transform
and classifies them into horizontal, vertical, and diagonal structural candidates.
"""

import math
from typing import List, Tuple
import cv2
import numpy as np
from models import WallCandidate

class WallDetector:
    def __init__(
        self,
        min_line_length: int = 30,
        max_line_gap: int = 15,
        hough_threshold: int = 40
    ):
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.hough_threshold = hough_threshold

    def detect_walls(self, binary_edges: np.ndarray) -> List[WallCandidate]:
        """
        Detects lines using Probabilistic Hough Transform and categorizes their orientation.
        """
        lines = cv2.HoughLinesP(
            binary_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        candidates: List[WallCandidate] = []
        if lines is None:
            return candidates

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)

            if length < self.min_line_length:
                continue

            angle_deg = math.degrees(math.atan2(abs(dy), abs(dx)))

            # Orientation Classification
            if angle_deg <= 15:
                orientation = "horizontal"
            elif angle_deg >= 75:
                orientation = "vertical"
            else:
                orientation = "diagonal"

            candidates.append(WallCandidate(
                start_point=(int(x1), int(y1)),
                end_point=(int(x2), int(y2)),
                orientation=orientation,
                length_px=float(length)
            ))

        return candidates
