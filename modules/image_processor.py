"""
SPAZE - Image Processor Pipeline
--------------------------------
Deterministic OpenCV image processing pipeline:
Grayscale -> Bilateral/Gaussian Denoising -> Otsu/Adaptive Thresholding ->
Canny Edge Detection -> Morphological Closing.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np

class ImageProcessor:
    def __init__(
        self,
        denoise_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 5
    ):
        self.denoise_kernel = denoise_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel_size = morph_kernel_size

    def process_pipeline(self, bgr_img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Runs the multi-stage OpenCV preprocessing pipeline.
        Returns a dictionary containing intermediate and final processed masks.
        """
        # 1. Grayscale conversion
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian / Bilateral filtering (preserves edges while eliminating scanner noise)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. Otsu & Adaptive Thresholding
        # Invert so walls/lines become foreground white (255) and empty space becomes black (0)
        _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Canny Edge Detection
        canny_edges = cv2.Canny(denoised, self.canny_low, self.canny_high)

        # 5. Morphological Closing (dilates then erodes to bridge door gaps and small architectural breaks)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        morphed = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 6. Distance Transform for room core region identification
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(morphed), cv2.DIST_L2, 5)

        return {
            "grayscale": gray,
            "denoised": denoised,
            "threshold": otsu_thresh,
            "canny_edges": canny_edges,
            "morph_closed": morphed,
            "dist_transform": dist_transform
        }
