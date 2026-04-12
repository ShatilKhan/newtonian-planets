# Newtonian Planets — Project Plan

## Goal
Build a series of Newtonian mechanics visualizations, starting with the Three-Body Problem.

---

## Stage 1: Three-Body Problem — 2D Animation (COMPLETE)

### Theory
- **Newton's Law of Gravitation:** `F = G * m₁ * m₂ / r²`
- **Equation of motion for body i:**
  `a_i = Σ_{j≠i} [ G * mⱼ * (rⱼ − rᵢ) / |rⱼ − rᵢ|³ ]`
- **Numerical integration:** 4th-order Runge-Kutta (RK4)
- **Softening parameter:** prevents singularity as r → 0

### Implementation
- File: `three_body.py`
- Initial conditions: Chenciner–Montgomery figure-8 periodic orbit
- Features: animated trails, body labels, time counter, energy readout (KE + PE + total E)
- Stack: Python, NumPy, Matplotlib

### Run
```bash
python3 three_body.py
```

---

## Stage 2: Chaotic Orbit + Energy Conservation (COMPLETE)

### Implementation
- File: `three_body_chaotic.py`
- Initial conditions: perturbed equilateral triangle with random velocity kick — produces chaotic bounded motion
- Layout: orbit animation (left) | total energy E(t) (top-right) | KE and PE (bottom-right)
- Energy drift at 10 000 steps with RK4 + softening=0.05, dt=0.002: **~0.1%**
- Seed-controlled: change `SEED` for a different chaotic trajectory

### Run
```bash
python3 three_body_chaotic.py
```

---

## Stage 3: 3D Chaotic Orbit — Enhanced (COMPLETE)

### Implementation
- File: `three_body_3d.py`
- Physics: identical to 2D but all vectors are (x, y, z)
- Initial conditions: perturbed equilateral triangle in xy-plane + random z-kick
- Camera: auto-rotates azimuth each frame (`ax.view_init`)
- **Speed slider**: `matplotlib.widgets.Slider`, 0.25× – 8×, steps of 0.25; controls how many simulation frames are consumed per animation frame
- **Units on all values**: AU (distance), M☉ (mass), yr (time), AU/yr (velocity), M☉·AU²/yr² (energy); G=1 normalised
- **Physics equations panel**: dedicated right column showing force law, equation of motion, energy definitions, integrator details, unit system, and live initial-state values
- Live body labels: name, mass, speed (|v| AU/yr), and (x,y,z) position
- Energy drift at 10 000 steps: **~0.000000%** (machine precision)

### Run
```bash
python3 three_body_3d.py
```

---

## Stage 4: Cinematic Visualization (COMPLETE)

### Implementation
- File: `three_body_beautiful.py`
- Gradient trails via `LineCollection` with per-segment RGBA (tail transparent → head bright)
- Glow effect: each trail drawn twice — wide+dim (halo) + thin+bright (core)
- Body glow: 4 concentric scatter layers per body (outer haze → hot core)
- Starfield: static scatter with exponential size distribution
- Neon palette: hot pink `#FF4D9E` / electric cyan `#00D4FF` / gold `#FFD166`
- Minimal UI: no axis ticks/borders, monospace labels, deep space background `#000010`
- Energy panel retained, styled to match dark theme

### Run
```bash
python3 three_body_beautiful.py
```

---

## Stage 5: Ideas for Next Steps (not started)
- [ ] Interactive initial conditions (click to place bodies)
- [ ] N-body generalization (more than 3 bodies)
- [ ] Save animation to MP4/GIF
