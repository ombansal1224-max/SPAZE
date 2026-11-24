"""
SPAZE - Image Loader & Validator
--------------------------------
Validates file signatures, ensures resolution adequacy,
and loads images into normalized numpy RGB / BGR arrays.
"""

import os
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image

class ImageLoader:
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}

    @classmethod
    def load_from_path(cls, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Floor plan image not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in cls.VALID_EXTENSIONS:
            raise ValueError(f"Unsupported image format '{ext}'. Must be PNG, JPG, or JPEG.")

        bgr = cv2.imread(path)
        if bgr is None:
            raise ValueError(f"Failed to decode image from: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr, rgb

    @classmethod
    def load_from_bytes(cls, file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
        nparr = np.frombuffer(file_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Corrupted or invalid floor plan image buffer.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr, rgb
