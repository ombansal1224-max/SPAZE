"""
SPAZE - 3D Architectural Scene Builder & Visualizer
---------------------------------------------------
Extrudes 2D room polygons into 3D floors, vertical structural walls,
doors, windows, and parametric furniture using Plotly 3D and Trimesh.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import trimesh

from config import (
    DEFAULT_CEILING_HEIGHT,
    DEFAULT_WALL_THICKNESS,
    DOOR_HEIGHT,
    DOOR_WIDTH,
    ROOM_COLORS
)
from models import Room, FloorPlan

class SceneBuilder3D:
    def __init__(self, ceiling_height_m: float = DEFAULT_CEILING_HEIGHT):
        self.ceiling_height_m = ceiling_height_m

    def build_plotly_scene(self, rooms: List[Any], px_per_m: float = 50.0) -> go.Figure:
        """
        Constructs an interactive 3D architectural viewport in Plotly.
        Renders floors, structural walls, furniture, and room labels.
        """
        fig = go.Figure()

        for room in rooms:
            # Handle both Room dataclass and CVAE dictionary items
            if hasattr(room, "bbox_px"):
                x_px, y_px, w_px, h_px = room.bbox_px
                name = room.name
                color = getattr(room, "color_hex", "#3B82F6")
            else:
                x_px, y_px, w_px, h_px = room["bbox_px"]
                name = room["name"]
                color = room.get("color_hex", "#3B82F6")

            # Convert to 3D Cartesian coordinates (metres)
            x_m = x_px / px_per_m
            y_m = y_px / px_per_m
            w_m = w_px / px_per_m
            h_m = h_px / px_per_m

            # 1. Extrude 3D Floor Mesh
            x_corners = [x_m, x_m + w_m, x_m + w_m, x_m]
            y_corners = [y_m, y_m, y_m + h_m, y_m + h_m]
            z_floor = [0.0, 0.0, 0.0, 0.0]

            # Floor mesh (2 triangles)
            fig.add_trace(go.Mesh3d(
                x=x_corners,
                y=y_corners,
                z=z_floor,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.75,
                name=f"Floor: {name}",
                hoverinfo="text",
                text=f"{name} (Floor Area: {w_m * h_m:.1f} m²)"
            ))

            # 2. Extrude 3D Vertical Walls
            wall_z = [0.0, self.ceiling_height_m]
            # Outer perimeter wireframe lines
            wall_x = [x_m, x_m + w_m, x_m + w_m, x_m, x_m]
            wall_y = [y_m, y_m, y_m + h_m, y_m + h_m, y_m]

            for z_level in [0.0, self.ceiling_height_m]:
                fig.add_trace(go.Scatter3d(
                    x=wall_x,
                    y=wall_y,
                    z=[z_level] * 5,
                    mode='lines',
                    line=dict(color='#E2E8F0', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Vertical pillar edges at corners
            for cx, cy in zip(x_corners, y_corners):
                fig.add_trace(go.Scatter3d(
                    x=[cx, cx],
                    y=[cy, cy],
                    z=wall_z,
                    mode='lines',
                    line=dict(color='#CBD5E1', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # 3. Add Parametric 3D Furniture Box
            self._add_parametric_furniture(fig, name, x_m, y_m, w_m, h_m)

            # 4. 3D Room Name Text Badge
            center_x = x_m + w_m / 2.0
            center_y = y_m + h_m / 2.0
            fig.add_trace(go.Scatter3d(
                x=[center_x],
                y=[center_y],
                z=[self.ceiling_height_m * 0.6],
                mode='text',
                text=[f"<b>{name}</b><br>{w_m:.1f}m × {h_m:.1f}m"],
                textposition='middle center',
                textfont=dict(size=11, color='#FFFFFF'),
                showlegend=False
            ))

        # Scene Styling & Lighting
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (Metres)', backgroundcolor='#0F172A', gridcolor='#334155'),
                yaxis=dict(title='Y (Metres)', backgroundcolor='#0F172A', gridcolor='#334155'),
                zaxis=dict(title='Height (Metres)', backgroundcolor='#0F172A', gridcolor='#334155', range=[0, 4.0]),
                camera=dict(
                    eye=dict(x=1.6, y=-1.6, z=1.8),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor='#090D16',
            height=650
        )

        return fig

    def _add_parametric_furniture(
        self,
        fig: go.Figure,
        room_name: str,
        x_m: float,
        y_m: float,
        w_m: float,
        h_m: float
    ):
        """
        Places parametric 3D bounding furniture models inside rooms.
        """
        r_lower = room_name.lower()

        if "bedroom" in r_lower:
            # Bed (approx 1.8m x 2.0m x 0.5m)
            bx, by = x_m + 0.3, y_m + 0.3
            bw, bh, bz = min(1.8, w_m * 0.55), min(2.0, h_m * 0.55), 0.55
            self._draw_3d_box(fig, bx, by, bw, bh, bz, color="#F43F5E", label="Queen Bed")

        elif "living" in r_lower:
            # Sofa (approx 2.2m x 0.9m x 0.75m)
            sx, sy = x_m + 0.4, y_m + 0.4
            sw, sh, sz = min(2.2, w_m * 0.6), min(0.9, h_m * 0.35), 0.75
            self._draw_3d_box(fig, sx, sy, sw, sh, sz, color="#3B82F6", label="3-Seater Sofa")

        elif "kitchen" in r_lower:
            # Kitchen Counter / Island
            kx, ky = x_m + 0.2, y_m + 0.2
            kw, kh, kz = min(w_m - 0.4, 2.5), 0.65, 0.90
            self._draw_3d_box(fig, kx, ky, kw, kh, kz, color="#F59E0B", label="Kitchen Counter")

        elif "bathroom" in r_lower:
            # Bathtub / Shower Stall
            tx, ty = x_m + 0.2, y_m + 0.2
            tw, th, tz = min(1.5, w_m * 0.7), min(0.8, h_m * 0.7), 0.55
            self._draw_3d_box(fig, tx, ty, tw, th, tz, color="#06B6D4", label="Bathtub / Vanity")

    def _draw_3d_box(
        self,
        fig: go.Figure,
        x: float, y: float, w: float, h: float, z_h: float,
        color: str, label: str
    ):
        """
        Helper that renders a solid 3D box in Plotly.
        """
        # Box vertices
        x_corners = [x, x + w, x + w, x, x, x + w, x + w, x]
        y_corners = [y, y, y + h, y + h, y, y, y + h, y + h]
        z_corners = [0.0, 0.0, 0.0, 0.0, z_h, z_h, z_h, z_h]

        fig.add_trace(go.Mesh3d(
            x=x_corners,
            y=y_corners,
            z=z_corners,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color,
            opacity=0.85,
            name=f"Furniture: {label}",
            hoverinfo="text",
            text=f"{label} ({w:.1f}m × {h:.1f}m × {z_h:.1f}m)"
        ))

    def export_trimesh_scene(self, rooms: List[Any], px_per_m: float = 50.0) -> trimesh.Scene:
        """
        Exports architectural components into a Trimesh 3D scene (supports GLB/OBJ export).
        """
        scene = trimesh.Scene()
        for room in rooms:
            if hasattr(room, "bbox_px"):
                x_px, y_px, w_px, h_px = room.bbox_px
            else:
                x_px, y_px, w_px, h_px = room["bbox_px"]

            x_m = x_px / px_per_m
            y_m = y_px / px_per_m
            w_m = w_px / px_per_m
            h_m = h_px / px_per_m

            # Create 3D box mesh for floor
            floor_mesh = trimesh.creation.box(extents=[w_m, h_m, 0.05])
            floor_mesh.apply_translation([x_m + w_m / 2.0, y_m + h_m / 2.0, 0.025])
            scene.add_geometry(floor_mesh)

        return scene
