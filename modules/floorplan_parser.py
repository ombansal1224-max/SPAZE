"""
SPAZE - Floor Plan Parser Orchestrator
--------------------------------------
Coordinates full end-to-end 2D floor plan parsing:
Image Ingestion -> Preprocessing -> Wall Detection -> Room Extraction -> Scaling.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np

from models import FloorPlan, ScaleInfo
from modules.image_loader import ImageLoader
from modules.image_processor import ImageProcessor
from modules.wall_detector import WallDetector
from modules.room_extractor import RoomExtractor

class FloorPlanParser:
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel: int = 5,
        hough_threshold: int = 40,
        min_line_len: int = 30
    ):
        self.processor = ImageProcessor(
            canny_low=canny_low,
            canny_high=canny_high,
            morph_kernel_size=morph_kernel
        )
        self.wall_detector = WallDetector(
            min_line_length=min_line_len,
            hough_threshold=hough_threshold
        )
        self.room_extractor = RoomExtractor()

    def parse(
        self,
        image_input,
        image_name: str = "floorplan.png",
        scale_info: ScaleInfo = None
    ) -> Tuple[FloorPlan, Dict[str, np.ndarray]]:
        """
        Parses floor plan from file path or raw bytes.
        Returns the populated FloorPlan dataclass and intermediate diagnostic imagery.
        """
        if scale_info is None:
            scale_info = ScaleInfo()

        if isinstance(image_input, str):
            bgr, rgb = ImageLoader.load_from_path(image_input)
        else:
            bgr, rgb = ImageLoader.load_from_bytes(image_input)

        h, w = bgr.shape[:2]

        # 1. OpenCV Processing Pipeline
        pipeline_masks = self.processor.process_pipeline(bgr)

        # 2. Structural Wall Detection
        walls = self.wall_detector.detect_walls(pipeline_masks["canny_edges"])

        # 3. Enclosed Room Extraction & Quality Scoring
        rooms = self.room_extractor.extract_rooms(pipeline_masks["morph_closed"], scale_info)

        total_area = sum(r.area_m2 for r in rooms)

        floorplan = FloorPlan(
            image_name=image_name,
            image_width=w,
            image_height=h,
            rooms=rooms,
            walls=walls,
            scale_info=scale_info,
            total_area_m2=total_area,
            metadata={"rooms_count": len(rooms), "walls_count": len(walls)}
        )

        # 4. Synthesize Annotated Overlay Image
        annotated_img = rgb.copy()
        for r in rooms:
            x, y, rw, rh = r.bbox_px
            # Draw semi-transparent bounding box
            overlay = annotated_img.copy()
            cv2.rectangle(overlay, (x, y), (x + rw, y + rh), (59, 130, 246), -1)
            cv2.addWeighted(overlay, 0.25, annotated_img, 0.75, 0, annotated_img)

            # Draw border and label
            cv2.rectangle(annotated_img, (x, y), (x + rw, y + rh), (16, 185, 129), 2)
            label = f"{r.name} ({r.area_m2:.1f} m2)"
            cv2.putText(annotated_img, label, (x + 8, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw detected walls
        for wall in walls:
            color = (0, 255, 255) if wall.orientation == "horizontal" else (255, 0, 255)
            cv2.line(annotated_img, wall.start_point, wall.end_point, color, 2)

        pipeline_masks["annotated_result"] = annotated_img
        pipeline_masks["original_rgb"] = rgb

        return floorplan, pipeline_masks
