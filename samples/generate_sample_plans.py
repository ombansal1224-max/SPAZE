"""
SPAZE - Sample Floor Plan Generator
-----------------------------------
Generates clean, realistic 2D architectural sample floor plan images
matching the bundled sample plan mentioned in the research report:
(800 x 600 px two-bedroom apartment plan with labeled walls and rooms).
"""

import os
import cv2
import numpy as np

out_dir = "c:/Users/om/.gemini/antigravity-ide/scratch/spaze-ai-architectural-layout/samples"
os.makedirs(out_dir, exist_ok=True)

def generate_sample_2bhk(filename="sample_2bhk_apartment.png"):
    w, h = 800, 600
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    wall_color = (15, 15, 15)
    wall_thickness = 8

    # 1. Outer Perimeter Walls
    cv2.rectangle(canvas, (50, 50), (750, 550), wall_color, wall_thickness)

    # 2. Interior Walls separating rooms cleanly
    # Central vertical partition
    cv2.line(canvas, (400, 50), (400, 550), wall_color, wall_thickness)

    # Left half: Living Room (Top) vs Kitchen (Bottom)
    cv2.line(canvas, (50, 310), (400, 310), wall_color, wall_thickness)

    # Right half: Master Bedroom (Top) vs (Bed 2 & Bath) (Bottom)
    cv2.line(canvas, (400, 270), (750, 270), wall_color, wall_thickness)

    # Right bottom: Bath vs Bed 2
    cv2.line(canvas, (560, 270), (560, 550), wall_color, wall_thickness)

    # 3. Add thin door swings (architectural convention: thin line + arc, solid wall frame)
    # Living Room door
    cv2.line(canvas, (390, 220), (370, 250), (100, 100, 100), 2)
    # Kitchen door
    cv2.line(canvas, (390, 340), (370, 370), (100, 100, 100), 2)
    # Bath door
    cv2.line(canvas, (550, 320), (530, 350), (100, 100, 100), 2)
    # Bed 2 door
    cv2.line(canvas, (410, 380), (430, 410), (100, 100, 100), 2)

    # 4. Text Room Labels
    cv2.putText(canvas, "LIVING ROOM", (120, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    cv2.putText(canvas, "KITCHEN", (160, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)
    cv2.putText(canvas, "MASTER BEDROOM", (480, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)
    cv2.putText(canvas, "BATH", (440, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2)
    cv2.putText(canvas, "BEDROOM 2", (610, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

    # 5. Dimension Scale Line (680 px = 13.6 m -> 50 px/m)
    cv2.line(canvas, (60, 575), (740, 575), (80, 80, 80), 2)
    cv2.line(canvas, (60, 570), (60, 580), (80, 80, 80), 2)
    cv2.line(canvas, (740, 570), (740, 580), (80, 80, 80), 2)
    cv2.putText(canvas, "SCALE: 680 px = 13.6 m (50 px/m)", (260, 570),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, canvas)
    print(f"[SUCCESS] Generated Sample Floor Plan: {out_path}")
    return out_path

if __name__ == "__main__":
    generate_sample_2bhk()
