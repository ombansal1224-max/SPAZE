"""
SPAZE - Export & File Utilities
-------------------------------
JSON serialization, Shapely polygon validation,
and formatted architectural metric exports.
"""

import json
from typing import Dict, Any
from models import FloorPlan

class ExportUtils:
    @staticmethod
    def export_floorplan_json(floorplan: FloorPlan, filepath: str = "floorplan_export.json") -> str:
        data = floorplan.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return filepath

    @staticmethod
    def get_summary_markdown(floorplan: FloorPlan) -> str:
        lines = [
            f"### 📐 Architectural Summary: {floorplan.image_name}",
            f"- **Total Usable Area:** `{floorplan.total_area_m2:.2f} m²`",
            f"- **Enclosed Rooms Identified:** `{len(floorplan.rooms)}`",
            f"- **Structural Walls Detected:** `{len(floorplan.walls)}`",
            f"- **Scale Metric:** `{floorplan.scale_info.px_per_metre:.1f} px/m` (Real distance: {floorplan.scale_info.real_distance_m:.1f} m)",
            "",
            "| ID | Room Name | Dimensions (W × H) | Area (m²) | Rectangularity (R) | Solidity (S) | Quality Score |",
            "|:---|:---|:---|:---|:---|:---|:---|"
        ]

        for r in floorplan.rooms:
            lines.append(
                f"| {r.id} | **{r.name}** | {r.width_m:.2f} m × {r.height_m:.2f} m | "
                f"{r.area_m2:.2f} m² | {r.rectangularity:.3f} | {r.solidity:.3f} | `{r.quality_score:.3f}` |"
            )

        return "\n".join(lines)
