# PIPO — 3D ASCII Holographic Assistant Core

**PIPO** is a minimalist, high-performance 3D ASCII holographic interface with natural fluid water-ball physics, Web Audio acoustic bandpass filtering, and 21-node dual-hand skeletal computer vision telemetry.

---

## Architecture & Features

### 1. Pristine Fibonacci Sphere Geometry (`public/orb.js`)
- **Uniform Distribution**: 1,400 points uniformly distributed via Golden Angle ($\approx 137.5^\circ$) Fibonacci lattice with constant base radius $R_0 = 230\text{px}$.
- **Fluid Micro-Ripples & Harmonic Perturbation**:
  $$\Delta r = R_0 \cdot \left[0.015 \cdot \sin(2\theta + 3\phi + t) + \text{smoothAudio} \cdot 0.08 \cdot \sin(4\theta + 3\phi + 2t)\right]$$
- **Strict Radial Bounds**: Radius strictly clamped to $[0.92 \cdot R_0, 1.08 \cdot R_0]$, maintaining a pristine spherical form without starfish, butterfly, or flat disk deformations.
- **Depth-of-Field (DoF)**: Character density and opacity strictly mapped from projected $Z$-depth (foreground `@`, `#`, `0` with high contrast, background `·`, `.` with soft opacity).

### 2. Acoustic Near-Field Filtering & Gating (`public/orb.js`)
- **Web Audio Biquad Bandpass Filter**: Isolates human voice ($250\text{ Hz} - 3500\text{ Hz}$, $Q = 0.7$) to eliminate ambient low-frequency rumbles and high-frequency hiss.
- **Adaptive Ambient Noise Tracking & SNR Gate**: Dynamic baseline estimation with a $2.5\times$ SNR gate so only direct microphone speech triggers displacement.
- **Low-Pass Audio Easing**: Gradual easing equation (`smoothAudio += (target - smoothAudio) * 0.10`) for organic, non-flickering responses.

### 3. 21-Point Skeletal Vision Sensor (`core/gesture_engine.py`)
- **Complete Skeletal Rendering**: Renders all 21 joint landmark nodes in bright yellow (`BGR: 0, 255, 255`) and connecting bone segments in cyan (`BGR: 255, 255, 0`).
- **Rotational Robustness & Persistence**: 4-frame coordinate persistence buffer and tuned detection/tracking confidence thresholds ($0.40$) to prevent tracking dropouts when hands tilt or rotate sideways.
- **Deterministic Telemetry Stream**: Broadcasts low-pass EMA filtered palm centroid coordinates, pinch states with hysteresis, depth scaling, dual-hand delta, and slap/flick momentum over FastAPI WebSockets (`/ws`).

### 4. Minimalist Monochrome UI (`public/index.html` & `public/style.css`)
- **Dynamic Expanding Segmented Toggles**: Segmented button groups with `flex: 1.65` expansion on active selection and cubic-bezier easing.
- **Recessed Capsule Sliders**: Matte neutral pill tracks with centered hairline indicator grooves.
- **Streamlined Footer**: Direct format `${count} HAND${count === 1 ? '' : 'S'}` (`0 HANDS`, `1 HAND`, `2 HANDS`).

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Webcam and microphone

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/QUROOOOO/My-assistant-_-.git
cd My-assistant-_-

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Run
```bash
python run.py
```
Open `http://localhost:8000` in your web browser.

---

## Gesture Interactions
| Gesture | Action |
| :--- | :--- |
| **No Hands (Idle)** | Slow continuous idle spin, gentle spring centering to viewport |
| **Open Palm (Hover)** | Halts auto-spin; parallax tilt follows palm centroid |
| **Single Pinch (Grab)** | Locks 3D spatial anchor; drag the sphere in X/Y/Z space |
| **Dual Pinch (Dual Grab)** | Midpoint translates the sphere; hand distance ratio zooms 3D scale |
| **Palm Flick / Slap** | Imparts rotational momentum with viscous friction decay |
