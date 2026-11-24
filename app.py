"""
SPAZE - AI-Powered Architectural Layout Generation System
---------------------------------------------------------
Tagline: Imagine. Generate. Inhabit.
B.Tech CSE Mini Project | Course: CS3ED13 - Gen AI | Medi-Caps University

An end-to-end Generative AI & Computer Vision application that turns
ordinary 2D floor plans into structured architectural data, CVAE-generated
layout alternatives, and interactive 3D WebGL visualizations.
"""

import os
import json
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Core SPAZE Modules
from config import (
    DEFAULT_PX_PER_METRE,
    DEFAULT_CEILING_HEIGHT,
    ROOM_COLORS,
    ROOM_TYPES
)
from models import FloorPlan, ScaleInfo, Room
from modules.floorplan_parser import FloorPlanParser
from cvae.cvae_model import CVAELayoutGenerator
from geometry3d.scene_builder import SceneBuilder3D
from utils.export_utils import ExportUtils

# ----------------------------------------------------
# 1. Page Configuration & Custom Architectural CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="SPAZE — AI-Powered Architectural Layout System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #090D16;
        color: #F8FAFC;
    }

    /* Top Brand Header */
    .spaze-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(59, 130, 246, 0.25);
        padding: 1.4rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .spaze-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .spaze-tagline {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .badge-course {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid #6366F1;
        color: #818CF8;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Stat Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.15);
    }

    .kpi-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.1;
    }

    .kpi-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 2. Header Bar
# ----------------------------------------------------
st.markdown("""
<div class="spaze-header">
    <div>
        <div class="spaze-title">🏛️ SPAZE</div>
        <div class="spaze-tagline">AI-Powered Architectural Layout Generation System — <i>Imagine. Generate. Inhabit.</i></div>
    </div>
    <div style="text-align: right;">
        <span class="badge-course">MEDICAPS UNIVERSITY | CS3ED13 - Gen AI</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 3. Sidebar Controls & Calibration
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/blueprint.png", width=64)
    st.title("Control Center")

    st.subheader("📁 Floor Plan Ingestion")
    input_choice = st.radio(
        "Select Plan Source:",
        ["⚡ Use Bundled 2-BHK Apartment Plan (800x600)", "📤 Upload Custom 2D Blueprint (PNG/JPG)"],
        index=0
    )

    uploaded_file = None
    sample_path = "samples/sample_2bhk_apartment.png"

    if "Upload Custom" in input_choice:
        uploaded_file = st.file_uploader("Upload floor plan image", type=["png", "jpg", "jpeg"])
    else:
        if os.path.exists(sample_path):
            st.success("✅ Sample 2-Bedroom Apartment loaded!")
        else:
            st.info("Generating sample floor plan...")

    st.markdown("---")
    st.subheader("📏 Metric Scale Calibration")
    real_dist_m = st.number_input("Known Wall Reference Length (Metres)", value=13.6, min_value=1.0, step=0.5)
    pixel_dist = st.number_input("Pixel Length on Image (Pixels)", value=680.0, min_value=50.0, step=10.0)
    calculated_px_m = pixel_dist / real_dist_m
    st.caption(f"📐 Calibrated Scale: **{calculated_px_m:.1f} pixels per metre**")

    st.markdown("---")
    st.subheader("⚙️ Vision & CVAE Settings")
    canny_thresh = st.slider("Canny Detection Sensitivity", 30, 200, (50, 150))
    morph_k = st.slider("Morphological Door-Gap Kernel", 3, 11, 5, step=2)
    cvae_temp = st.slider("CVAE Generation Temperature (Diversity)", 0.2, 1.5, 0.85, 0.05)


# ----------------------------------------------------
# 4. Parsing Orchestration
# ----------------------------------------------------
parser = FloorPlanParser(
    canny_low=canny_thresh[0],
    canny_high=canny_thresh[1],
    morph_kernel=morph_k
)
scale_info = ScaleInfo(
    pixel_distance=pixel_dist,
    real_distance_m=real_dist_m,
    px_per_metre=calculated_px_m
)

# Determine input source
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    plan_name = uploaded_file.name
    floorplan, masks = parser.parse(file_bytes, image_name=plan_name, scale_info=scale_info)
else:
    # Use sample plan
    if not os.path.exists(sample_path):
        from samples.generate_sample_plans import generate_sample_2bhk
        generate_sample_2bhk()

    floorplan, masks = parser.parse(sample_path, image_name="sample_2bhk_apartment.png", scale_info=scale_info)


# ----------------------------------------------------
# 5. Top KPI Summary Dashboard
# ----------------------------------------------------
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Usable Area</div>
        <div class="kpi-value">{floorplan.total_area_m2:.1f} <span style="font-size: 1.1rem; color: #94A3B8;">m²</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Enclosed Rooms Identified</div>
        <div class="kpi-value" style="color: #A78BFA;">{len(floorplan.rooms)}</div>
    </div>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Structural Walls Detected</div>
        <div class="kpi-value" style="color: #34D399;">{len(floorplan.walls)}</div>
    </div>
    """, unsafe_allow_html=True)

with col_k4:
    avg_quality = np.mean([r.quality_score for r in floorplan.rooms]) if floorplan.rooms else 0.0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Mean Quality Score</div>
        <div class="kpi-value" style="color: #FBBF24;">{avg_quality:.2f} <span style="font-size: 1.1rem; color: #94A3B8;">/ 1.0</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ----------------------------------------------------
# 6. Five Tabbed Workspaces
# ----------------------------------------------------
tabs = st.tabs([
    "🔍 1. 2D Floor Plan Processing",
    "📊 2. Dimensions & Quality Scores",
    "🧠 3. CVAE Generative Layout Alternatives",
    "🏛️ 4. Interactive 3D Architectural Scene",
    "📝 5. Room Editor & JSON Export"
])


# ====================================================
# TAB 1: 2D Floor Plan Processing (Computer Vision Inspection)
# ====================================================
with tabs[0]:
    st.markdown("### 🖼️ Computer Vision Processing Stages")
    st.write("Deterministic OpenCV extraction pipeline: Grayscale conversion, Otsu thresholding, Canny edge detection, morphological closing (bridging door gaps), and contour hierarchy identification.")

    sub_col1, sub_col2 = st.columns([1, 1], gap="medium")

    with sub_col1:
        st.markdown("#### Input Blueprint & Intermediate Edge Masks")
        view_mode = st.selectbox(
            "Select Processing Stage to Inspect:",
            ["Original Blueprint", "Grayscale & Bilateral Denoised", "Otsu Inverted Threshold", "Canny Edge Detection", "Morphological Closed (Door-Gaps Bridged)"]
        )

        if view_mode == "Original Blueprint":
            st.image(masks["original_rgb"], caption="Original Input Floor Plan", use_column_width=True)
        elif view_mode == "Grayscale & Bilateral Denoised":
            st.image(masks["denoised"], caption="Stage 1: Bilateral Filter (Noise Removed, Edges Preserved)", use_column_width=True, clamp=True)
        elif view_mode == "Otsu Inverted Threshold":
            st.image(masks["threshold"], caption="Stage 2: Otsu Binary Inverted Mask", use_column_width=True, clamp=True)
        elif view_mode == "Canny Edge Detection":
            st.image(masks["canny_edges"], caption="Stage 3: Canny Edge Boundaries", use_column_width=True, clamp=True)
        else:
            st.image(masks["morph_closed"], caption="Stage 4: Morphological Closing (Doorways sealed for contour extraction)", use_column_width=True, clamp=True)

    with sub_col2:
        st.markdown("#### Final Annotated Semantic Architectural Plan")
        st.image(masks["annotated_result"], caption="Detected Rooms (Green/Blue) and Structural Wall Segments (Yellow/Magenta)", use_column_width=True)


# ====================================================
# TAB 2: Dimensions & Quality Analytics
# ====================================================
with tabs[1]:
    st.markdown("### 📐 Extracted Room Dimensions & Quality Heuristics")
    st.write("Quality Score Formula from SPAZE Report: `Quality Score = 0.45 * R + 0.35 * S + 0.20 * A`")
    st.caption("• **R (Rectangularity):** Ratio of room contour area to minimum bounding box area | • **S (Solidity):** Ratio of contour area to convex hull area | • **A (Aspect Ratio):** Proportional reward for aspect ratios between 0.5 and 2.0")

    table_data = []
    for r in floorplan.rooms:
        table_data.append({
            "Room ID": r.id,
            "Room Name": r.name,
            "Width (m)": f"{r.width_m:.2f} m",
            "Height (m)": f"{r.height_m:.2f} m",
            "Area (m²)": f"{r.area_m2:.2f} m²",
            "Rectangularity (R)": r.rectangularity,
            "Solidity (S)": r.solidity,
            "Aspect Ratio (A)": r.aspect_ratio,
            "Quality Score": f"⭐ {r.quality_score:.3f}"
        })

    st.dataframe(table_data, use_container_width=True)


# ====================================================
# TAB 3: Generative AI (CVAE) Layout Alternatives
# ====================================================
with tabs[2]:
    st.markdown("### 🧠 Generative AI: Conditional Variational Autoencoder (CVAE)")
    st.write("Generates multiple diverse, constraint-satisfied architectural layout alternatives from the learned continuous latent space conditioned on the identified room taxonomy.")

    gen_col1, gen_col2 = st.columns([1, 2], gap="large")

    with gen_col1:
        st.markdown("#### CVAE Latent Sampling")
        st.write("Conditioning on room set: `[x, y, w, h, type_id]`")
        num_alts = st.slider("Number of Alternatives to Sample", 2, 4, 3)
        resample_btn = st.button("🔄 Generate New CVAE Layout Alternatives", use_container_width=True)

        st.info("💡 The CVAE encodes the room constraints and draws diverse samples from the continuous latent space, preserving overall area while rearranging room circulation.")

    with gen_col2:
        cvae_engine = CVAELayoutGenerator()
        if "cvae_alternatives" not in st.session_state or resample_btn:
            st.session_state["cvae_alternatives"] = cvae_engine.generate_alternatives(
                base_rooms=floorplan.rooms,
                canvas_width=floorplan.image_width,
                canvas_height=floorplan.image_height,
                num_alternatives=num_alts,
                temperature=cvae_temp
            )

        alt_tabs = st.tabs([f"Alternative Proposal #{i+1}" for i in range(len(st.session_state["cvae_alternatives"]))])

        for idx, alt_rooms in enumerate(st.session_state["cvae_alternatives"]):
            with alt_tabs[idx]:
                alt_canvas = np.ones((floorplan.image_height, floorplan.image_width, 3), dtype=np.uint8) * 245

                for r in alt_rooms:
                    rx, ry, rw, rh = r["bbox_px"]
                    # Draw room block
                    cv2.rectangle(alt_canvas, (rx, ry), (rx + rw, ry + rh), (200, 220, 255), -1)
                    cv2.rectangle(alt_canvas, (rx, ry), (rx + rw, ry + rh), (59, 130, 246), 2)
                    cv2.putText(alt_canvas, f"{r['name']}", (rx + 8, ry + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2)
                    cv2.putText(alt_canvas, f"{r['area_m2']} m2", (rx + 8, ry + 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

                st.image(alt_canvas, caption=f"CVAE Alternative #{idx+1} — Diversity Score: {alt_rooms[0]['diversity_score']}", use_column_width=True)


# ====================================================
# TAB 4: Interactive 3D Architectural Scene
# ====================================================
with tabs[3]:
    st.markdown("### 🏛️ Interactive 3D Architectural Visualization (Plotly / WebGL)")
    st.write("2D room polygons extruded into solid 3D floors, vertical perimeter walls (2.8m ceiling), and parametric furniture models.")

    scene_choice = st.radio(
        "Choose Layout to Render in 3D:",
        ["Original Extracted 2D Floor Plan", "CVAE Alternative Proposal #1"],
        horizontal=True
    )

    builder3d = SceneBuilder3D(ceiling_height_m=DEFAULT_CEILING_HEIGHT)

    if "Original" in scene_choice:
        fig3d = builder3d.build_plotly_scene(floorplan.rooms, px_per_m=scale_info.px_per_metre)
    else:
        alt_rooms = st.session_state.get("cvae_alternatives", [floorplan.rooms])[0]
        fig3d = builder3d.build_plotly_scene(alt_rooms, px_per_m=scale_info.px_per_metre)

    st.plotly_chart(fig3d, use_container_width=True)


# ====================================================
# TAB 5: Room Editor & JSON Export
# ====================================================
with tabs[4]:
    st.markdown("### 📝 Human-in-the-Loop Room Editor & Data Export")
    st.write("Review and customize automatically detected labels, adjust dimensions, and export the structured floor plan data contract.")

    col_e1, col_e2 = st.columns([1, 1], gap="medium")

    with col_e1:
        st.markdown("#### Rename & Edit Detected Rooms")
        for room in floorplan.rooms:
            new_name = st.text_input(f"Room {room.id} Label:", value=room.name, key=f"room_rename_{room.id}")
            room.name = new_name

    with col_e2:
        st.markdown("#### Structured JSON Data Contract")
        json_data = floorplan.to_dict()
        st.json(json_data)

        json_str = json.dumps(json_data, indent=4)
        st.download_button(
            label="💾 Download Structured FloorPlan JSON",
            data=json_str,
            file_name=f"{floorplan.image_name.split('.')[0]}_spaze_export.json",
            mime="application/json",
            use_container_width=True
        )
