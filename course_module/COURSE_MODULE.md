# Course Module
# AI for Exoplanet Habitability — From Newtonian Mechanics to Machine Learning

**Module code:** ASTRO-AI 401
**Level:** Upper undergraduate / early graduate
**Duration:** 12 weeks (one semester) · 36 lecture hours + 36 lab hours
**Prerequisites:**
- Classical mechanics (Lagrangian/Hamiltonian preferred)
- Linear algebra & multivariable calculus
- Python programming (NumPy, basic matplotlib)
- Introductory probability / statistics

---

## 1 · Module Overview

This module builds a complete pipeline from the **Newtonian N-body problem** to **AI-driven assessment of exoplanet habitability**. Students will learn how the orbital dynamics of a planet around its host star (and within multi-planet systems) constrain whether it can sustain liquid water, and they will train ML models to predict habitability from observational catalogues and from N-body simulations.

The module is structured around three pillars:

1. **Physics** — Newtonian gravity, the chaotic three-body problem, orbital stability, the circumstellar habitable zone (CHZ).
2. **Data** — Real exoplanet catalogues (NASA Exoplanet Archive, Kepler/TESS, PHL-HEC) and N-body simulation data.
3. **AI / ML** — Classical ML for classification, deep learning for transit detection, physics-informed neural networks for orbital surrogates, symbolic regression for rediscovering physical laws.

A capstone project asks students to build a **dynamically-stable habitability classifier** that combines the SPOCK stability metric with PHL-EC habitability features.

---

## 2 · Learning Outcomes

By the end of the module, students will be able to:

| # | Outcome |
|---|---|
| LO1 | Derive and integrate the equations of motion for the N-body gravitational problem using RK4 / symplectic integrators |
| LO2 | Compute the circumstellar habitable zone for a given stellar luminosity and explain its physical limits |
| LO3 | Apply orbital-stability indicators (Hill stability, MEGNO, Lyapunov time, AMD) to multi-planet systems |
| LO4 | Query and clean exoplanet datasets (NASA Exoplanet Archive, PHL-HEC, Kepler/TESS) for ML pipelines |
| LO5 | Train, validate, and interpret classical ML classifiers (RF / XGBoost / SVM) for exoplanet habitability |
| LO6 | Implement a deep-learning pipeline for transit detection (AstroNet-style CNN on Kepler light curves) |
| LO7 | Build a physics-informed / Hamiltonian neural network surrogate for an N-body integrator |
| LO8 | Use symbolic regression (PySR / AI Feynman) to recover Kepler's third law from numerical data |
| LO9 | Critically discuss the limitations and biases of habitability indices (ESI, PHI, CDHS, CEESA) |
| LO10 | Communicate ML-for-astrophysics results in a short conference-style paper |

---

## 3 · Weekly Schedule

### Week 1 — Foundations of Orbital Mechanics
- Newton's law of gravitation; two-body problem; Kepler's laws
- Conservation laws (energy, angular momentum, Laplace–Runge–Lenz vector)
- **Lab:** Implement the two-body problem in NumPy; verify Kepler T² ∝ a³ numerically

### Week 2 — The N-body Problem and Numerical Integration
- General N-body equations
- Numerical integrators: Euler, Verlet, RK4, symplectic (leapfrog, WHFast)
- Energy/momentum conservation diagnostics
- **Lab:** Three-body simulation with RK4; reproduce the figure-8 periodic orbit (Chenciner & Montgomery, 2000)

### Week 3 — The Restricted Three-Body Problem & Lagrange Points
- The R3BP framework, the Jacobi constant, zero-velocity surfaces
- Lagrange points L1–L5 and their stability
- Real-world examples: Sun-Earth L2 (JWST), Trojan asteroids
- **Lab:** Map zero-velocity curves; integrate test particles near L4/L5

### Week 4 — Orbital Stability and Chaos
- Lyapunov exponents and Lyapunov time
- MEGNO chaos indicator (Cincotta & Simó, 2000)
- Hill stability (Marchal & Bozis, 1982; Gladman, 1993)
- AMD-stability (Laskar & Petit, 2017)
- **Lab:** Compute MEGNO for randomly-sampled three-planet systems using `REBOUND`

### Week 5 — Habitable Zones and Climate Drivers
- The CHZ: Kasting et al. (1993); Kopparapu et al. (2013, 2014)
- Eccentricity, obliquity, and Milankovitch cycles
- Tidal locking around M-dwarfs (Barnes 2017)
- **Lab:** Compute CHZ inner/outer edges across the HR diagram

### Week 6 — Habitability Indices
- ESI — Earth Similarity Index (Schulze-Makuch et al. 2011)
- PHI — Planetary Habitability Index
- CDHS — Cobb–Douglas Habitability Score (Bora, Saha et al. 2016)
- CEESA — Constant Elasticity Earth Similarity Approach (Saha et al. 2018)
- Critique: arbitrary thresholds, geocentric bias
- **Lab:** Compute ESI/PHI on the PHL-HEC catalogue; visualise distributions

### Week 7 — Exoplanet Detection Methods and Datasets
- Transit photometry, radial velocity, microlensing, direct imaging, astrometry
- Kepler / K2 / TESS / PLATO / Gaia / HWO
- NASA Exoplanet Archive TAP queries
- **Lab:** Pull the NASA PSCompPars table; build a feature matrix

### Week 8 — Classical ML for Habitability
- Random Forest, XGBoost, SVM, KNN
- Class imbalance (potentially-habitable planets are <1% of confirmed)
- SMOTE / class-weighting; cross-validation; SHAP feature importance
- Saha et al. (2018) and Basak et al. (2020) reproduced
- **Lab:** Train an XGBoost habitability classifier on PHL-EC; report ROC-AUC

### Week 9 — Deep Learning for Transit Detection
- CNN architectures for time-series light curves
- AstroNet (Shallue & Vanderburg, 2018) and ExoMiner (Valizadegan et al. 2022)
- Data augmentation: phase-folding, normalisation, injection-recovery
- **Lab:** Build a 1-D CNN on Kepler light curves (Kaggle dataset); evaluate

### Week 10 — Physics-Informed Neural Networks for Dynamics
- Hamiltonian Neural Networks (Greydanus et al. 2019)
- Lagrangian Neural Networks (Cranmer et al. 2020)
- Symplectic networks (SympNets, Jin et al. 2020)
- "Newton vs the Machine" — Breen et al. (2020) on the chaotic three-body
- **Lab:** Train an HNN on two-body data; compare long-term energy drift vs. RK4

### Week 11 — Symbolic Regression and Interpretable Astrophysics
- AI Feynman (Udrescu & Tegmark 2020)
- PySR (Cranmer 2023) and "Discovering Symbolic Models with Inductive Biases"
- Lemos et al. (2022): rediscovering orbital mechanics with ML
- **Lab:** Use PySR to recover Kepler's third law from simulated planetary data

### Week 12 — Capstone Presentations + Future Directions
- Student presentations of capstone projects
- Discussion: PLATO, HWO, Roman, Ariel — the next decade
- Open problems: atmosphere retrieval with ML; Bayesian uncertainty for habitability

---

## 4 · Capstone Project

**Title:** *Dynamically-Stable Habitability — A Joint Classifier*

**Brief.** Combine the **SPOCK** orbital-stability classifier (Tamayo et al. 2020) with **PHL-EC** habitability features. Train a model that predicts whether a hypothetical planet is *both* dynamically stable for ≥ 10⁹ orbits *and* potentially habitable (T_eq, ESI, mass/radius constraints).

**Pipeline:**
1. Sample 10 000 synthetic three-planet systems in REBOUND.
2. Run SPOCK to assign a stability probability to each.
3. Cross-join with PHL-EC features for the middle planet.
4. Train an XGBoost + small MLP ensemble to predict the joint label.
5. Report SHAP feature importances and ROC-AUC.
6. Write a 4-page conference-style paper (template provided).

**Deliverables:** code repo, paper PDF, 10-min presentation.

---

## 5 · Assessment

| Component | Weight |
|---|---|
| Weekly lab notebooks (12 × 2.5%) | 30% |
| Mid-term: 2-page reproduction of one paper (e.g. Saha 2018) | 20% |
| Capstone project (code + paper + presentation) | 40% |
| Class participation (seminar discussions) | 10% |

---

## 6 · Required Resources

### Textbooks
- Murray & Dermott, *Solar System Dynamics*, Cambridge UP, 1999.
- Ivezić, Connolly, VanderPlas & Gray, *Statistics, Data Mining, and Machine Learning in Astronomy*, Princeton UP, 2nd ed. 2019. https://www.astroml.org/
- Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press, 2016.

### Software
- Python 3.11+, NumPy, SciPy, scikit-learn, XGBoost, PyTorch
- `REBOUND` (N-body) — https://github.com/hannorein/rebound
- `lightkurve` (Kepler/TESS) — https://github.com/lightkurve/lightkurve
- `exoplanet` (PyMC) — https://github.com/exoplanet-dev/exoplanet
- `SPOCK` (stability classifier) — https://github.com/dtamayo/spock
- `PySR` (symbolic regression) — https://github.com/MilesCranmer/PySR

### Datasets
- NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu/
- PHL Habitable Exoplanets Catalog — https://phl.upr.edu/hwc
- Kepler / TESS via MAST — https://archive.stsci.edu/
- Kaggle "Exoplanet Hunting in Deep Space" — https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

---

## 7 · Recommended Lectures & Videos

- **Sara Seager — MIT 12.425 Extrasolar Planets** (OCW): https://ocw.mit.edu/courses/12-425-extrasolar-planets-physics-and-detection-techniques-fall-2013/
- **Cool Worlds Lab (David Kipping)**: https://www.youtube.com/@CoolWorldsLab
- **PBS Space Time**: https://www.youtube.com/@pbsspacetime
- **Veritasium — three-body problem**: https://www.youtube.com/watch?v=et7XvBenEo8
- **HarvardX — Super-Earths and Life (Sasselov)**: https://www.edx.org/learn/astronomy/harvard-university-super-earths-and-life
- **NeurIPS ML4Physical Sciences Workshop**: https://ml4physicalsciences.github.io/
- **ML4Astro at ICML**: https://ml4astro.github.io/icml2022/

---

## 8 · Reading List (15 Core Papers)

1. Kasting, Whitmire & Reynolds (1993), *Icarus* 101, 108. DOI:10.1006/icar.1993.1010
2. Kopparapu et al. (2013), arXiv:1301.6674
3. Schulze-Makuch et al. (2011), *Astrobiology* 11, 1041. DOI:10.1089/ast.2010.0592
4. Shallue & Vanderburg (2018) — AstroNet, arXiv:1712.05044
5. Valizadegan et al. (2022) — ExoMiner, arXiv:2111.10009
6. Pearson, Palafox & Griffith (2018), arXiv:1706.04319
7. Saha et al. (2018), arXiv:1712.01040
8. Bora, Saha et al. (2016) — CDHS, arXiv:1604.01722
9. Basak et al. (2020), arXiv:1805.08810
10. Breen et al. (2020) — Newton vs the Machine, arXiv:1910.07291
11. Greydanus et al. (2019) — HNN, arXiv:1906.01563
12. Cranmer et al. (2020) — LNN, arXiv:2003.04630
13. Cranmer et al. (2020) — Symbolic Models, arXiv:2006.11287
14. Lemos et al. (2022) — Rediscovering orbital mechanics, arXiv:2202.02306
15. Laskar & Petit (2017) — AMD-stability, arXiv:1703.07125
