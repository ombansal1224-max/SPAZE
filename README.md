# 🏛️ SPAZE — AI-Powered Architectural Layout Generation System

> **Tagline:** *Imagine. Generate. Inhabit.*  
> **Course:** CS3ED13 - Generative AI | **Department:** Computer Science & Engineering | **Institute:** Medi-Caps University, Indore

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Generative AI](https://img.shields.io/badge/Generative_AI-CVAE-FF6F00?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_3D-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📌 Project Overview & Vision
**SPAZE** is an end-to-end AI-assisted architectural engineering application that turns ordinary 2D floor-plan blueprints into structured architectural data, generative CVAE layout alternatives, and interactive 3D WebGL scenes.

Architects and interior planners spend hours manually redrawing plans and testing room arrangements. SPAZE automates this chain by coupling **classical computer vision**, **computational geometry**, a **Conditional Variational Autoencoder (CVAE)**, and **interactive 3D mesh extrusion**.

---

## 🚀 Key Technological Capabilities

1. **Deterministic 2D Floor Plan Ingestion (OpenCV):**
   - Bilateral & Gaussian noise reduction, Otsu adaptive thresholding, and Canny edge detection.
   - Morphological closing to bridge physical door gaps and architectural line breaks.
   - Probabilistic Hough Line Transform to classify horizontal, vertical, and diagonal structural wall candidates.
   - Contour hierarchy analysis with Douglas-Peucker polygon simplification to extract enclosed room cavities.

2. **Computational Geometry & Metric Scaling:**
   - Real-world metric calibration: Converts pixel distances to metres (e.g. $680\text{ px} = 13.6\text{ m} \implies 50\text{ px/m}$).
   - Computes physical width ($m$), height ($m$), and usable floor area ($m^2$).
   - **Quality Score Formula:**
     $$\text{Quality Score} = 0.45 \times R + 0.35 \times S + 0.20 \times A$$
     - $R$: Rectangularity ($\text{contour area} / \text{bounding box area}$)
     - $S$: Solidity ($\text{contour area} / \text{convex hull area}$)
     - $A$: Aspect Ratio Score (rewards natural proportions between 0.5 and 2.0)

3. **Generative AI: Conditional Variational Autoencoder (CVAE):**
   - Learns a continuous latent space of architectural room vectors $[x, y, w, h]$ conditioned on room taxonomy (`type_id`).
   - Generates multiple diverse layout variations for the same room set while enforcing boundary constraints and avoiding overlapping geometry.

4. **Interactive 3D Architectural Scene Construction (Plotly / Trimesh):**
   - Automatically extrudes 2D room polygons into solid 3D floors with room-specific architectural palette colors.
   - Extrudes vertical structural perimeter walls to standard residential height ($2.8\text{ m}$).
   - Populates rooms with parametric 3D furniture models (queen beds in bedrooms, sofas in living rooms, counters in kitchens, tubs/vanities in bathrooms).
   - Allows full browser orbit, rotate, pan, zoom, and 3D export.

5. **Human-in-the-Loop Room Editor & Data Contract:**
   - Allows architects to customize room names, adjust bounds, and download a standardized `FloorPlan.json` data contract.

---

## 🛠️ Project Architecture

```
spaze-ai-architectural-layout/
├── app.py                      # Main Streamlit web application & tabbed interface
├── config.py                   # Central constants, room taxonomy, and color palettes
├── models.py                   # Dataclass schemas: FloorPlan, Room, WallCandidate, ScaleInfo
├── RUN_PROJECT.bat             # 1-Click launcher for Windows
├── modules/
│   ├── image_loader.py         # File signature validator and image array loader
│   ├── image_processor.py      # Multi-stage OpenCV preprocessing pipeline
│   ├── wall_detector.py        # Probabilistic Hough Line wall detection
│   ├── room_extractor.py       # Contour hierarchy, quality scoring & scaling
│   └── floorplan_parser.py     # Orchestrates full end-to-end 2D parsing
├── cvae/
│   └── cvae_model.py           # Conditional Variational Autoencoder layout generator
├── geometry3d/
│   └── scene_builder.py        # Trimesh & Plotly 3D extruded architectural scene builder
├── utils/
│   └── export_utils.py         # JSON serialization and Markdown summaries
└── samples/
    ├── generate_sample_plans.py# Generator for clean architectural benchmark plans
    └── sample_2bhk_apartment.png # Bundled sample 2-BHK modern apartment blueprint
```

---

## 💻 How to Run

### Method 1: 1-Click Runner (Easiest)
Simply double-click:
👉 **`RUN_PROJECT.bat`**

### Method 2: From Terminal
```powershell
cd c:\Users\om\.gemini\antigravity-ide\scratch\spaze-ai-architectural-layout
python -m streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**
