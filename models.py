"""
SPAZE - Core Data Models & Schemas
----------------------------------
Dataclass schema matching SPAZE specifications:
FloorPlan, Room, WallCandidate, and ScaleInfo.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class WallCandidate:
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    orientation: str  # 'horizontal', 'vertical', or 'diagonal'
    length_px: float
    thickness_px: float = 8.0

@dataclass
class ScaleInfo:
    pixel_distance: float = 680.0
    real_distance_m: float = 13.6
    px_per_metre: float = 50.0

@dataclass
class Room:
    id: int
    name: str
    type_id: int
    bbox_px: Tuple[int, int, int, int]  # (x, y, w, h)
    polygon_px: List[Tuple[int, int]]
    area_m2: float
    width_m: float
    height_m: float
    rectangularity: float
    solidity: float
    aspect_ratio: float
    quality_score: float
    color_hex: str = "#3B82F6"
    confidence: float = 0.95

@dataclass
class FloorPlan:
    image_name: str
    image_width: int
    image_height: int
    rooms: List[Room] = field(default_factory=list)
    walls: List[WallCandidate] = field(default_factory=list)
    scale_info: ScaleInfo = field(default_factory=ScaleInfo)
    total_area_m2: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "dimensions_px": {"width": self.image_width, "height": self.image_height},
            "scale_info": {
                "px_per_metre": self.scale_info.px_per_metre,
                "real_distance_m": self.scale_info.real_distance_m
            },
            "total_area_m2": round(self.total_area_m2, 2),
            "rooms": [
                {
                    "id": r.id,
                    "name": r.name,
                    "type_id": r.type_id,
                    "bbox_px": r.bbox_px,
                    "dimensions_m": {"width": round(r.width_m, 2), "height": round(r.height_m, 2)},
                    "area_m2": round(r.area_m2, 2),
                    "quality_metrics": {
                        "rectangularity": round(r.rectangularity, 3),
                        "solidity": round(r.solidity, 3),
                        "aspect_ratio": round(r.aspect_ratio, 3),
                        "quality_score": round(r.quality_score, 3)
                    }
                }
                for r in self.rooms
            ],
            "walls_count": len(self.walls)
        }
