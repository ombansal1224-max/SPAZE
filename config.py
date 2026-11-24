"""
SPAZE - Configuration & Constants
---------------------------------
Central configurations, room type taxonomy, color schemes,
and default architectural heuristics.
"""

# Room Type Taxonomy
ROOM_TYPES = {
    0: "Living Room",
    1: "Master Bedroom",
    2: "Bedroom",
    3: "Kitchen",
    4: "Bathroom",
    5: "Balcony",
    6: "Dining Area",
    7: "Hallway / Corrdior"
}

ROOM_COLORS = {
    "Living Room": "#3B82F6",       # Blue
    "Master Bedroom": "#8B5CF6",    # Purple
    "Bedroom": "#EC4899",           # Pink
    "Kitchen": "#F59E0B",           # Amber
    "Bathroom": "#06B6D4",          # Cyan
    "Balcony": "#10B981",           # Emerald
    "Dining Area": "#F97316",       # Orange
    "Hallway / Corrdior": "#6B7280" # Gray
}

# 3D Architectural Dimensions (in Metres)
DEFAULT_CEILING_HEIGHT = 2.80   # Standard residential room height
DEFAULT_WALL_THICKNESS = 0.20   # 20cm external / structural wall
INTERIOR_WALL_THICKNESS = 0.12  # 12cm partition wall
DOOR_WIDTH = 0.90               # Standard 90cm door
DOOR_HEIGHT = 2.10              # Standard 2.1m door height
WINDOW_SILL_HEIGHT = 0.90       # Standard 90cm window sill
WINDOW_HEIGHT = 1.20            # Standard 1.2m window opening

# Default Scale Calibration
DEFAULT_PX_PER_METRE = 50.0     # Default 50 pixels = 1.0 metre (e.g. 680px = 13.6m)

# Quality Score Weights (Formula from SPAZE Research Report)
# Quality Score = 0.45 * R + 0.35 * S + 0.20 * A
WEIGHT_RECTANGULARITY = 0.45
WEIGHT_SOLIDITY = 0.35
WEIGHT_ASPECT_RATIO = 0.20

# CVAE Latent Dimensions
CVAE_LATENT_DIM = 8
CVAE_FEATURE_DIM = 4   # [x, y, w, h]
CVAE_NUM_CLASSES = 8   # One-hot room type conditioning
