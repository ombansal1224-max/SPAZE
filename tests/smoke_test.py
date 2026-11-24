"""
SPAZE - Automated Smoke Test Suite
----------------------------------
Validates image loading, OpenCV preprocessing, wall detection,
room extraction, metric scaling, CVAE generation, and 3D scene building.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.floorplan_parser import FloorPlanParser
from cvae.cvae_model import CVAELayoutGenerator
from geometry3d.scene_builder import SceneBuilder3D
from models import ScaleInfo

class SpazeSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = "samples/sample_2bhk_apartment.png"
        if not os.path.exists(cls.sample_path):
            from samples.generate_sample_plans import generate_sample_2bhk
            generate_sample_2bhk()

    def test_01_floorplan_parser(self):
        parser = FloorPlanParser()
        scale = ScaleInfo(pixel_distance=680.0, real_distance_m=13.6, px_per_metre=50.0)
        floorplan, masks = parser.parse(self.sample_path, scale_info=scale)

        self.assertIsNotNone(floorplan)
        self.assertGreater(len(floorplan.rooms), 0, "Should detect at least 1 room")
        self.assertGreater(floorplan.total_area_m2, 0.0, "Total area should be positive")
        self.assertIn("annotated_result", masks)

    def test_02_quality_score(self):
        parser = FloorPlanParser()
        floorplan, _ = parser.parse(self.sample_path)
        for room in floorplan.rooms:
            self.assertGreaterEqual(room.quality_score, 0.0)
            self.assertLessEqual(room.quality_score, 1.0)
            self.assertGreater(room.width_m, 0.0)
            self.assertGreater(room.height_m, 0.0)

    def test_03_cvae_generator(self):
        parser = FloorPlanParser()
        floorplan, _ = parser.parse(self.sample_path)
        cvae = CVAELayoutGenerator()
        alts = cvae.generate_alternatives(
            floorplan.rooms,
            floorplan.image_width,
            floorplan.image_height,
            num_alternatives=3
        )
        self.assertEqual(len(alts), 3, "Should generate 3 alternatives")
        self.assertEqual(len(alts[0]), len(floorplan.rooms), "Room count should be conserved")

    def test_04_3d_scene_builder(self):
        parser = FloorPlanParser()
        floorplan, _ = parser.parse(self.sample_path)
        builder = SceneBuilder3D()
        fig = builder.build_plotly_scene(floorplan.rooms)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0, "3D figure should contain meshes and traces")

if __name__ == "__main__":
    unittest.main()
