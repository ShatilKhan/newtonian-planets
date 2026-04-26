"""
Builds a short conference-style PDF paper using ReportLab.

Title: Toward AI-Driven Habitability Prediction from Orbital Dynamics:
       A Course-Module Position Paper

Output: course_module/paper.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem,
)
from reportlab.platypus.flowables import HRFlowable
import os

OUT = os.path.join(os.path.dirname(__file__), 'paper.pdf')

# ─── Styles ──────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()

title_style = ParagraphStyle(
    'TitleStyle', parent=ss['Title'],
    fontName='Helvetica-Bold', fontSize=17, leading=21,
    alignment=TA_CENTER, spaceAfter=6, textColor=black,
)
subtitle_style = ParagraphStyle(
    'SubtitleStyle', parent=ss['Normal'],
    fontName='Helvetica-Oblique', fontSize=11, leading=14,
    alignment=TA_CENTER, spaceAfter=12, textColor=HexColor('#444444'),
)
author_style = ParagraphStyle(
    'AuthorStyle', parent=ss['Normal'],
    fontName='Helvetica', fontSize=10, leading=12,
    alignment=TA_CENTER, spaceAfter=4,
)
affil_style = ParagraphStyle(
    'AffilStyle', parent=ss['Normal'],
    fontName='Helvetica-Oblique', fontSize=9, leading=11,
    alignment=TA_CENTER, spaceAfter=14, textColor=HexColor('#555555'),
)
h1 = ParagraphStyle(
    'H1', parent=ss['Heading1'],
    fontName='Helvetica-Bold', fontSize=12, leading=15,
    spaceBefore=12, spaceAfter=6, textColor=HexColor('#102a43'),
)
h2 = ParagraphStyle(
    'H2', parent=ss['Heading2'],
    fontName='Helvetica-Bold', fontSize=10.5, leading=13,
    spaceBefore=8, spaceAfter=4, textColor=HexColor('#243b53'),
)
body = ParagraphStyle(
    'Body', parent=ss['Normal'],
    fontName='Helvetica', fontSize=9.5, leading=13,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
body_indent = ParagraphStyle(
    'BodyIndent', parent=body, leftIndent=12,
)
abstract_style = ParagraphStyle(
    'Abstract', parent=body, leftIndent=18, rightIndent=18,
    fontSize=9, leading=12, textColor=HexColor('#222222'),
)
mono = ParagraphStyle(
    'Mono', parent=body, fontName='Courier', fontSize=8.5, leading=11,
)
ref_style = ParagraphStyle(
    'Ref', parent=body, fontSize=8.5, leading=11, leftIndent=14, firstLineIndent=-14,
    spaceAfter=2,
)

# ─── Build ───────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.0*cm, bottomMargin=2.0*cm,
    title='AI-Driven Habitability Prediction from Orbital Dynamics',
    author='ASTRO-AI 401',
)

flow = []

# ── Title block ──
flow.append(Paragraph(
    'Toward AI-Driven Habitability Prediction<br/>from Orbital Dynamics',
    title_style))
flow.append(Paragraph(
    'A Position Paper for the ASTRO-AI 401 Course Module',
    subtitle_style))
flow.append(Paragraph('Course Author Team — ASTRO-AI 401', author_style))
flow.append(Paragraph(
    'Department of Astrophysics &amp; Computer Science · Draft v0.1',
    affil_style))
flow.append(HRFlowable(width='100%', thickness=0.6, color=HexColor('#cccccc')))
flow.append(Spacer(1, 8))

# ── Abstract ──
flow.append(Paragraph('<b>Abstract.</b>', body))
flow.append(Paragraph(
    "The chaotic Newtonian N-body problem governs the long-term orbital "
    "configuration of every planetary system, and orbital dynamics in turn "
    "set hard constraints on whether any given world can sustain liquid "
    "water. We outline a unified framework for using modern machine "
    "learning to predict exoplanet habitability directly from orbital and "
    "stellar parameters. We review the canonical habitable-zone formulations "
    "of Kasting et al. (1993) and Kopparapu et al. (2013); the existing "
    "habitability indices (ESI, PHI, CDHS, CEESA); and the rapidly maturing "
    "ML literature spanning gradient-boosted classifiers (Saha et al. 2018), "
    "deep transit detectors (Shallue &amp; Vanderburg 2018; ExoMiner 2022), "
    "physics-informed neural networks (Greydanus 2019; Cranmer 2020), and "
    "ML-based stability classifiers (SPOCK; Tamayo 2020). We propose a "
    "joint <i>dynamically-stable habitability</i> metric that fuses SPOCK-style "
    "long-horizon stability with PHL-EC habitability features, and we sketch "
    "the corresponding course module and capstone project.",
    abstract_style))
flow.append(Spacer(1, 6))

# ── 1. Introduction ──
flow.append(Paragraph('1. Introduction', h1))
flow.append(Paragraph(
    "Of the more than 5 800 confirmed exoplanets in the NASA Exoplanet "
    "Archive (as of 2026), only a small fraction lie within their host "
    "star's circumstellar habitable zone (CHZ). Even fewer have been "
    "shown to be <i>dynamically</i> stable on geological timescales. "
    "Yet long-term stability is a precondition for life: a planet whose "
    "eccentricity oscillates on Myr timescales may experience climate "
    "swings incompatible with biosphere persistence. The classical "
    "Newtonian N-body problem is analytically intractable for N ≥ 3 "
    "(Poincaré, 1890), and direct symplectic integration of every "
    "candidate system remains expensive. Machine learning offers two "
    "complementary tools: (i) cheap surrogates for the dynamics, and "
    "(ii) data-driven classifiers that map observed orbital and stellar "
    "features to habitability labels.",
    body))

# ── 2. Foundational Physics ──
flow.append(Paragraph('2. Foundational Physics', h1))

flow.append(Paragraph('2.1 Equations of Motion', h2))
flow.append(Paragraph(
    "For a system of N point masses interacting only through Newtonian "
    "gravity, the acceleration of body <i>i</i> is",
    body))
flow.append(Paragraph(
    "&nbsp;&nbsp;&nbsp;&nbsp;a<sub>i</sub> = Σ<sub>j ≠ i</sub> "
    "G · m<sub>j</sub> · (r<sub>j</sub> − r<sub>i</sub>) / |r<sub>j</sub> − r<sub>i</sub>|<sup>3</sup>",
    mono))
flow.append(Paragraph(
    "Energy E = KE + PE is conserved analytically; numerical integrators "
    "(RK4, Verlet, leapfrog, WHFast) preserve it to varying degrees. "
    "Symplectic schemes maintain bounded energy error indefinitely and are "
    "the standard for long-horizon planetary dynamics.",
    body))

flow.append(Paragraph('2.2 Habitable Zones', h2))
flow.append(Paragraph(
    "The canonical CHZ (Kasting et al. 1993; Kopparapu et al. 2013) is the "
    "annular region around a star in which an Earth-like rocky planet can "
    "support liquid surface water. Inner edge: runaway/moist greenhouse. "
    "Outer edge: maximum CO₂ greenhouse before condensation. The CHZ scales "
    "approximately as r<sub>HZ</sub> ∝ √(L<sub>★</sub>/L<sub>☉</sub>).",
    body))

flow.append(Paragraph('2.3 Stability Criteria', h2))
flow.append(Paragraph(
    "Long-term habitability also requires orbital stability. Standard "
    "indicators include Hill stability (Marchal &amp; Bozis 1982; Gladman "
    "1993), the MEGNO chaos indicator (Cincotta &amp; Simó 2000), Lyapunov "
    "exponents, and AMD-stability (Laskar &amp; Petit 2017). The SPOCK "
    "classifier (Tamayo et al. 2020, <i>PNAS</i> 117, 18194) uses a "
    "gradient-boosted model to predict long-horizon stability of compact "
    "multi-planet systems orders of magnitude faster than direct N-body.",
    body))

# ── 3. Habitability Indices ──
flow.append(Paragraph('3. Habitability Indices', h1))
flow.append(Paragraph(
    "Several scalar indices attempt to summarise habitability:",
    body))

idx_table_data = [
    ['Index', 'Reference', 'Inputs'],
    ['ESI',   'Schulze-Makuch et al. 2011',  'R, ρ, V_esc, T_surf'],
    ['PHI',   'Schulze-Makuch et al. 2011',  'substrate, energy, chemistry, liquid'],
    ['CDHS',  'Bora, Saha et al. 2016',      'R, M, T_eq, T_s — Cobb–Douglas form'],
    ['CEESA', 'Saha et al. 2018',            'CES production-function generalisation'],
]
idx_table = Table(idx_table_data, colWidths=[2.4*cm, 5.5*cm, 8.0*cm])
idx_table.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), 'Helvetica', 8.5),
    ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 8.5),
    ('TEXTCOLOR', (0,0), (-1,0), HexColor('#102a43')),
    ('BACKGROUND', (0,0), (-1,0), HexColor('#eef3f8')),
    ('GRID', (0,0), (-1,-1), 0.3, HexColor('#bbbbbb')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
flow.append(idx_table)
flow.append(Spacer(1, 6))
flow.append(Paragraph(
    "All four are anchored on Earth (ESI(Earth) ≡ 1), inheriting an obvious "
    "geocentric bias. Their utility is largely as <i>features</i> for "
    "downstream ML rather than as standalone verdicts.",
    body))

# ── 4. Machine Learning Methods ──
flow.append(Paragraph('4. Machine-Learning Methods', h1))

flow.append(Paragraph('4.1 Classification of Confirmed Planets', h2))
flow.append(Paragraph(
    "Saha et al. (2018) and Basak et al. (2020) train Random Forests, "
    "XGBoost, KNN, and SVMs on the PHL Exoplanet Catalog (PHL-EC) to "
    "predict the discrete habitability class (non-habitable, mesoplanet, "
    "psychroplanet, etc.). Class imbalance (potentially-habitable planets "
    "are well under 1% of the catalogue) is handled via SMOTE or class "
    "weighting. Reported test ROC-AUC scores exceed 0.95, although the "
    "evaluation is intrinsically circular because habitability labels are "
    "themselves derived from the same features.",
    body))

flow.append(Paragraph('4.2 Deep Learning for Transit Detection', h2))
flow.append(Paragraph(
    "AstroNet (Shallue &amp; Vanderburg 2018, <i>AJ</i> 155, 94) is a 1-D "
    "CNN trained on Kepler light curves; it discovered Kepler-90i and "
    "Kepler-80g. ExoMiner (Valizadegan et al. 2022, <i>ApJ</i> 926, 120) "
    "extends this with explainable per-feature attention and validated 301 "
    "new exoplanets. These detection models feed downstream habitability "
    "pipelines by enlarging the input catalogue.",
    body))

flow.append(Paragraph('4.3 Physics-Informed Neural Networks', h2))
flow.append(Paragraph(
    "Hamiltonian Neural Networks (HNNs; Greydanus, Dzamba &amp; Yosinski "
    "2019), Lagrangian Neural Networks (LNNs; Cranmer et al. 2020), and "
    "Symplectic Networks (SympNets; Jin et al. 2020) bake conservation "
    "laws into the architecture. Breen et al. (2020) showed that a deep "
    "network trained on chaotic three-body trajectories can extrapolate "
    "orders of magnitude faster than Brutus while preserving global "
    "structure. For habitability work, such surrogates make Monte-Carlo "
    "stability sweeps tractable.",
    body))

flow.append(Paragraph('4.4 Symbolic Regression', h2))
flow.append(Paragraph(
    "AI Feynman (Udrescu &amp; Tegmark 2020) and PySR (Cranmer 2023) "
    "rediscover analytic laws from numerical data. Lemos et al. (2022, "
    "arXiv:2202.02306) recover Newtonian gravity from solar-system "
    "trajectories — a powerful pedagogical tool for the course module's "
    "Week 11 lab.",
    body))

# ── 5. Proposed Joint Metric ──
flow.append(Paragraph('5. Proposed Joint Metric', h1))
flow.append(Paragraph(
    "We propose a <b>Dynamically-Stable Habitability Score (DSHS)</b> "
    "defined as the product of (i) the SPOCK-predicted probability of "
    "stability over 10⁹ orbits and (ii) a learned habitability score from "
    "an XGBoost classifier trained on PHL-EC features:",
    body))
flow.append(Paragraph(
    "&nbsp;&nbsp;&nbsp;&nbsp;DSHS = P<sub>SPOCK</sub>(stable | system) "
    "× P<sub>XGB</sub>(habitable | r<sub>p</sub>, m<sub>p</sub>, T<sub>eq</sub>, S/S<sub>⊕</sub>, …)",
    mono))
flow.append(Paragraph(
    "Calibration is performed against the existing PHL-HEC catalogue and "
    "augmented by 10 000 synthetic three-planet REBOUND systems. SHAP "
    "feature-importance plots quantify which orbital/stellar parameters "
    "drive each component independently.",
    body))

# ── 6. Course Integration ──
flow.append(Paragraph('6. Course Integration', h1))
flow.append(Paragraph(
    "The DSHS pipeline is the capstone project of ASTRO-AI 401. The 12-week "
    "module steps students from two-body integration (Week 1) through "
    "REBOUND-based stability sweeps (Weeks 2–4), CHZ computation (Week 5), "
    "the PHL catalogues and indices (Week 6), classical and deep ML for "
    "habitability and transit detection (Weeks 7–9), HNN/SympNet surrogates "
    "(Week 10), and symbolic regression of orbital laws (Week 11). The "
    "final week is devoted to capstone presentations.",
    body))
flow.append(Paragraph(
    "Open-source tools: REBOUND, SPOCK, lightkurve, the exoplanet PyMC "
    "package, PySR. All datasets are public (NASA Exoplanet Archive, MAST, "
    "PHL-HEC, Kaggle Kepler).",
    body))

# ── 7. Outlook ──
flow.append(Paragraph('7. Outlook', h1))
flow.append(Paragraph(
    "PLATO (launch 2026), the Roman Space Telescope (2027), and the "
    "Habitable Worlds Observatory (2040s) will deliver orders of magnitude "
    "more candidate worlds. Combined with JWST and Ariel atmospheric "
    "spectroscopy, the next decade will saturate ML pipelines with high-"
    "dimensional, heterogeneous data. Models that respect underlying "
    "Newtonian conservation laws — and that quantify their own uncertainty "
    "— will be the credible tools for ranking which worlds deserve the "
    "limited follow-up time of flagship observatories.",
    body))

# ── References ──
flow.append(Paragraph('References', h1))

refs = [
    "Basak, S., Saha, S., et al. (2020). Habitability classification of exoplanets: a machine learning insight. <i>EPJ Special Topics</i>. arXiv:1805.08810.",
    "Bora, K., Saha, S., et al. (2016). CD-HPF: a new habitability score via the Cobb–Douglas habitability production function. arXiv:1604.01722.",
    "Breen, P. G., Foley, C. N., Boekholt, T., Portegies Zwart, S. (2020). Newton vs. the machine: solving the chaotic three-body problem using deep neural networks. <i>MNRAS</i> 494, 2465. arXiv:1910.07291.",
    "Cincotta, P. M., Simó, C. (2000). Simple tools to study global dynamics in non-axisymmetric galactic potentials – I. <i>A&amp;AS</i> 147, 205.",
    "Cranmer, M. D., Greydanus, S., et al. (2020). Lagrangian Neural Networks. arXiv:2003.04630.",
    "Cranmer, M. D., et al. (2020). Discovering symbolic models from deep learning with inductive biases. <i>NeurIPS</i>. arXiv:2006.11287.",
    "Greydanus, S., Dzamba, M., Yosinski, J. (2019). Hamiltonian Neural Networks. <i>NeurIPS</i>. arXiv:1906.01563.",
    "Kasting, J. F., Whitmire, D. P., Reynolds, R. T. (1993). Habitable zones around main-sequence stars. <i>Icarus</i> 101, 108. DOI:10.1006/icar.1993.1010.",
    "Kopparapu, R. K., et al. (2013). Habitable zones around main-sequence stars: new estimates. arXiv:1301.6674.",
    "Laskar, J., Petit, A. C. (2017). AMD-stability and the classification of planetary systems. <i>A&amp;A</i> 605, A72. arXiv:1703.07125.",
    "Lemos, P., Jeffrey, N., Cranmer, M., Ho, S., Battaglia, P. (2022). Rediscovering orbital mechanics with machine learning. arXiv:2202.02306.",
    "Pearson, K. A., Palafox, L., Griffith, C. A. (2018). Searching for exoplanets using artificial intelligence. <i>MNRAS</i> 474, 478. arXiv:1706.04319.",
    "Saha, S., et al. (2018). Theoretical validation of potential habitability via analytical and boosted tree methods. <i>Astron. Comput.</i> 23, 141. arXiv:1712.01040.",
    "Schulze-Makuch, D., et al. (2011). A two-tiered approach to assessing the habitability of exoplanets. <i>Astrobiology</i> 11, 1041. DOI:10.1089/ast.2010.0592.",
    "Shallue, C. J., Vanderburg, A. (2018). Identifying exoplanets with deep learning: a five-planet resonant chain around Kepler-90. <i>AJ</i> 155, 94. arXiv:1712.05044.",
    "Tamayo, D., et al. (2020). Predicting the long-term stability of compact multiplanet systems. <i>PNAS</i> 117, 18194. arXiv:2007.06521.",
    "Udrescu, S.-M., Tegmark, M. (2020). AI Feynman: a physics-inspired method for symbolic regression. <i>Sci. Adv.</i> 6, eaay2631. arXiv:1905.11481.",
    "Valizadegan, H., et al. (2022). ExoMiner: a highly accurate and explainable deep learning classifier. <i>ApJ</i> 926, 120. arXiv:2111.10009.",
]
for r in refs:
    flow.append(Paragraph(r, ref_style))

# Build
doc.build(flow)
print(f"Wrote: {OUT}")
