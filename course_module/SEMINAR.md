# Seminar
# *Can a Neural Network Tell Us Where Life Could Live?*
## AI-Driven Habitability Prediction from Orbital Dynamics

**Format:** 90-minute graduate seminar (60 min talk + 20 min live demo + 10 min Q&A)
**Audience:** Physics, astronomy, and CS graduate students
**Prerequisites:** undergraduate mechanics; basic ML literacy

---

## Abstract

The chaotic three-body problem has frustrated astronomers since Poincaré. Yet a planet's long-term habitability depends *precisely* on the chaotic dance of itself, its star, and its sibling planets. This seminar shows how modern machine learning — from gradient-boosted classifiers to Hamiltonian neural networks — is being used to predict whether an exoplanet, given only its orbital configuration and stellar context, could host liquid water. We tour the key methods (AstroNet, ExoMiner, SPOCK, HNNs), demonstrate a live PyTorch surrogate of an N-body integrator, and end with the open problem: **can an AI ever say with confidence that a world is habitable?**

---

## Outline (90 min)

### Part 1 — The Physics (20 min)
- The Newtonian N-body problem (and why N=3 is hard)
- The circumstellar habitable zone — Kasting (1993), Kopparapu (2013)
- Why orbital dynamics matter for life: eccentricity drives climate, resonances drive stability, tidal locking decides whether one face boils

### Part 2 — The Data (15 min)
- Kepler / TESS / PLATO / Gaia
- The NASA Exoplanet Archive and the PHL Habitable Exoplanets Catalog
- Habitability indices (ESI, PHI, CDHS, CEESA) — what they measure and what they hide

### Part 3 — The Models (25 min)
- **AstroNet** (Shallue & Vanderburg 2018) — a CNN finds Kepler-90i
- **ExoMiner** (Valizadegan 2022) — explainable transit vetting
- **Saha et al. (2018)** — gradient boosting on PHL-EC
- **SPOCK** (Tamayo et al. 2020) — ML-based stability classification
- **HNN / LNN / SympNets** — physics-informed surrogates that conserve energy
- **PySR** — symbolic regression that *rediscovers* Kepler

### Part 4 — Live Demo (20 min)
1. Run the bundled `three_body_3d.py` — chaotic N-body with energy conservation
2. Inject a "habitable" candidate; show ESI computation
3. Train a small XGBoost classifier on a slice of PHL-EC
4. Show an HNN trained on two-body data conserving energy 1000× better than RK4 over long horizons

### Part 5 — Q&A (10 min)
- Open problems: uncertainty quantification, atmospheric retrieval with ML, generative models for unseen planets

---

## Demo File Map

| File | Purpose |
|---|---|
| `three_body.py` | Figure-8 periodic orbit (Chenciner & Montgomery) — beautiful baseline |
| `three_body_chaotic.py` | Chaotic 2D system + energy plot |
| `three_body_3d.py` | 3D chaotic system, units, equations panel, speed slider |
| (planned) `habitability_xgboost.ipynb` | Train classifier on PHL-EC features |
| (planned) `hnn_two_body.ipynb` | HNN surrogate vs. RK4 long-horizon energy drift |

---

## Key Visuals to Prepare

1. CHZ map across the HR diagram
2. PHL-HEC distribution of ESI vs. orbital period
3. AstroNet ROC curve (reproduced from Shallue & Vanderburg 2018)
4. SPOCK feature-importance bar chart (TRAPPIST-1 example)
5. HNN energy-drift plot (HNN flat line vs. RK4 cumulative drift)
6. PySR equation discovery: input data → recovered T² ∝ a³

---

## Suggested Discussion Prompts

- *Habitability indices like ESI rank Earth = 1.0 by definition. What does that imply about generalisation to truly alien chemistries?*
- *If an HNN can predict three-body trajectories far longer than RK4, does it "understand" gravity? Or has it just memorised?*
- *NASA's HWO will image fewer than ~25 exo-Earths in its lifetime. How do we use ML responsibly when N is that small?*
- *Can a model trained on Kepler photometry transfer to PLATO data? What about TESS to JWST atmospheres?*

---

## Pre-Seminar Reading (one item)

Either:
- Shallue & Vanderburg (2018), *AJ* 155, 94 — arXiv:1712.05044  *(deep learning entry point)*
- Saha et al. (2018), *Astron. Comput.* 23, 141 — arXiv:1712.01040  *(habitability ML entry point)*

---

## Speaker Bio Slot

[Your bio here — research interests, group, contact]
