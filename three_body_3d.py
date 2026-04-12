"""
Three-Body Problem — 3D Chaotic Orbit
======================================

Features
--------
  • 3D orbit with auto-rotating camera
  • Energy conservation plots (total E, KE, PE)
  • Physics equations panel on screen
  • Speed slider  (0.25× – 8× real-time)

Unit system (G = 1 normalised)
-------------------------------
  Length  : AU   (astronomical unit, ~1.496 × 10¹¹ m)
  Mass    : M☉   (solar mass,        ~1.989 × 10³⁰ kg)
  Time    : yr   (year,              ~3.156 × 10⁷  s)
  Velocity: AU/yr
  Energy  : M☉ · AU² / yr²
  G = 4π² ≈ 39.48 AU³ M☉⁻¹ yr⁻² in SI, normalised to 1 here.

Integrator : 4th-order Runge-Kutta (RK4), dt = 0.002 yr
Softening  : ε = 0.05 AU  (prevents singularity at r → 0)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─── Simulation parameters ───────────────────────────────────────────────────
G         = 1.0           # normalised gravitational constant
SOFTENING = 0.05          # AU
DT        = 0.002         # yr
N_STEPS   = 10000
TRAIL_LEN = 500
SEED      = 42

# ─── Initial Conditions ──────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

MASSES = rng.uniform(0.8, 1.5, size=3)   # M☉

R      = 1.0              # AU — initial triangle radius
angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
xy     = np.column_stack([
    R * np.cos(angles) + rng.uniform(-0.15, 0.15, 3),
    R * np.sin(angles) + rng.uniform(-0.15, 0.15, 3),
])
z_kick   = rng.uniform(-0.4, 0.4, size=(3, 1))   # AU — breaks planar symmetry
POS_INIT = np.hstack([xy, z_kick])                 # shape (3, 3)

v_scale  = np.sqrt(G * MASSES.sum() / R) * 0.4    # AU/yr
VEL_INIT = rng.uniform(-v_scale, v_scale, size=(3, 3))
VEL_INIT -= (MASSES[:, None] * VEL_INIT).sum(axis=0) / MASSES.sum()  # zero CoM

print(f"Masses : {MASSES.round(3)} M☉")
print(f"Seed   : {SEED}")

# ─── Physics ─────────────────────────────────────────────────────────────────

def accelerations(pos, masses):
    """
    Gravitational acceleration on each body (AU/yr²).

        aᵢ = Σⱼ≠ᵢ  G·mⱼ·(rⱼ − rᵢ) / (|rⱼ − rᵢ|² + ε²)^(3/2)
    """
    n   = len(masses)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d  = pos[j] - pos[i]
            r2 = np.dot(d, d) + SOFTENING**2
            acc[i] += G * masses[j] * d / r2**1.5
    return acc


def derivatives(state, masses):
    n = len(masses)
    return np.concatenate([state[n:], accelerations(state[:n], masses)])


def rk4(state, masses, dt):
    k1 = derivatives(state,              masses)
    k2 = derivatives(state + 0.5*dt*k1,  masses)
    k3 = derivatives(state + 0.5*dt*k2,  masses)
    k4 = derivatives(state +     dt*k3,  masses)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def total_energy(pos, vel, masses):
    """
    E = KE + PE

    KE = Σᵢ  ½·mᵢ·|vᵢ|²                      [M☉·AU²/yr²]
    PE = −Σᵢ<ⱼ  G·mᵢ·mⱼ / |rᵢⱼ|              [M☉·AU²/yr²]
    """
    n  = len(masses)
    ke = sum(0.5 * masses[i] * np.dot(vel[i], vel[i]) for i in range(n))
    pe = sum(
        -G * masses[i] * masses[j]
        / np.sqrt(np.dot(pos[j]-pos[i], pos[j]-pos[i]) + SOFTENING**2)
        for i in range(n) for j in range(i+1, n)
    )
    return ke + pe, ke, pe

# ─── Integrate ───────────────────────────────────────────────────────────────
print("Integrating...")
state      = np.concatenate([POS_INIT, VEL_INIT])
trajectory = np.zeros((N_STEPS, 3, 3))   # (step, body, xyz) [AU]
velocities = np.zeros((N_STEPS, 3, 3))   # (step, body, xyz) [AU/yr]
energy_tot = np.zeros(N_STEPS)
energy_ke  = np.zeros(N_STEPS)
energy_pe  = np.zeros(N_STEPS)
time_arr   = np.arange(N_STEPS) * DT     # [yr]

trajectory[0] = POS_INIT
velocities[0] = VEL_INIT
energy_tot[0], energy_ke[0], energy_pe[0] = total_energy(POS_INIT, VEL_INIT, MASSES)

for step in range(1, N_STEPS):
    state = rk4(state, MASSES, DT)
    trajectory[step] = state[:3]
    velocities[step] = state[3:]
    e, ke, pe        = total_energy(state[:3], state[3:], MASSES)
    energy_tot[step] = e
    energy_ke[step]  = ke
    energy_pe[step]  = pe

E0    = energy_tot[0]
drift = abs(energy_tot[-1] - E0) / abs(E0) * 100
print(f"Done.  E₀={E0:.4f} M☉·AU²/yr²   drift={drift:.6f}%")

# ─── Axis limits ─────────────────────────────────────────────────────────────
pad  = 0.4
span = max(
    trajectory[:,:,0].max() - trajectory[:,:,0].min(),
    trajectory[:,:,1].max() - trajectory[:,:,1].min(),
    trajectory[:,:,2].max() - trajectory[:,:,2].min(),
) / 2 + pad
cx, cy, cz = (trajectory[:,:,i].mean() for i in range(3))

# ─── Theme ───────────────────────────────────────────────────────────────────
DARK   = '#0d1117'
PANEL  = '#111820'
GRID   = '#1e2a38'
TEXT   = '#c9d1d9'
COLORS = ['#e63946', '#457b9d', '#2a9d8f']
NAMES  = ['Body 1', 'Body 2', 'Body 3']

# ─── Figure & layout ─────────────────────────────────────────────────────────
#
#   ┌─────────────────┬───────────┬─────────────────┐
#   │                 │  Total E  │                 │
#   │   3D Orbit      │           │   Equations     │
#   │                 ├───────────┤   Panel         │
#   │                 │  KE / PE  │                 │
#   └─────────────────┴───────────┴─────────────────┘
#   │              Speed Slider                     │
#   └───────────────────────────────────────────────┘

fig = plt.figure(figsize=(16, 8), facecolor=DARK)
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.04, right=0.97,
    top=0.91,  bottom=0.12,   # bottom gap for slider
    hspace=0.50, wspace=0.30,
    width_ratios=[2.2, 1.3, 1.3],
)

ax3d  = fig.add_subplot(gs[:, 0], projection='3d')
ax_et = fig.add_subplot(gs[0, 1])
ax_kp = fig.add_subplot(gs[1, 1])
ax_eq = fig.add_subplot(gs[:, 2])   # equations panel

# ── 3D orbit axes ─────────────────────────────────────────────────────────────
ax3d.set_facecolor(PANEL)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor(GRID)
ax3d.tick_params(colors=TEXT, labelsize=7)
ax3d.set_xlabel('x  (AU)',  color=TEXT, fontsize=8, labelpad=2)
ax3d.set_ylabel('y  (AU)',  color=TEXT, fontsize=8, labelpad=2)
ax3d.set_zlabel('z  (AU)',  color=TEXT, fontsize=8, labelpad=2)
ax3d.set_title('3D Orbital Trajectories', color=TEXT, fontsize=10, pad=8)
ax3d.set_xlim(cx-span, cx+span)
ax3d.set_ylim(cy-span, cy+span)
ax3d.set_zlim(cz-span, cz+span)

# ── Energy axes ───────────────────────────────────────────────────────────────
for ax in (ax_et, ax_kp):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('time  (yr)', color=TEXT, fontsize=8)

ax_et.set_title('Total Energy  E = KE + PE', color=TEXT, fontsize=10, pad=6)
ax_et.set_ylabel('E  (M☉·AU²/yr²)', color=TEXT, fontsize=7.5)
ax_et.axhline(E0, color='#888', lw=0.8, linestyle=':', label=f'E₀ = {E0:.3f}')
ax_et.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
ax_et.set_xlim(0, N_STEPS * DT)
e_margin = max(abs(energy_tot - E0).max() * 1.5, 1e-6)
ax_et.set_ylim(E0 - e_margin, E0 + e_margin)
e_line,   = ax_et.plot([], [], color='#f4a261', lw=1.2)
drift_txt = ax_et.text(0.97, 0.06, '', transform=ax_et.transAxes,
                        color='#f4a261', fontsize=7, ha='right')

ax_kp.set_title('Kinetic & Potential Energy', color=TEXT, fontsize=10, pad=6)
ax_kp.set_ylabel('energy  (M☉·AU²/yr²)', color=TEXT, fontsize=7.5)
ax_kp.set_xlim(0, N_STEPS * DT)
ax_kp.set_ylim(energy_pe.min() * 1.1, energy_ke.max() * 1.1)
ke_line, = ax_kp.plot([], [], color='#e63946', lw=1.0, label='KE')
pe_line, = ax_kp.plot([], [], color='#457b9d', lw=1.0, label='PE')
ax_kp.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# ── Physics equations panel ───────────────────────────────────────────────────
ax_eq.set_facecolor(PANEL)
ax_eq.set_xticks([])
ax_eq.set_yticks([])
for sp in ax_eq.spines.values():
    sp.set_edgecolor(GRID)
ax_eq.set_title('Physics', color=TEXT, fontsize=10, pad=6)

EQ_TEXT = (
    "Gravitational force\n"
    "  F = G·m₁·m₂ / r²\n"
    "\n"
    "Equation of motion\n"
    "  aᵢ = Σⱼ≠ᵢ  G·mⱼ·(rⱼ−rᵢ)\n"
    "            ──────────────\n"
    "            (|rⱼ−rᵢ|²+ε²)^(3/2)\n"
    "\n"
    "Total energy (conserved)\n"
    "  E = KE + PE\n"
    "\n"
    "  KE = Σᵢ ½·mᵢ·|vᵢ|²\n"
    "\n"
    "  PE = −Σᵢ<ⱼ G·mᵢ·mⱼ/|rᵢⱼ|\n"
    "\n"
    "Integrator\n"
    "  Runge-Kutta 4th order\n"
    "  dt = 0.002 yr\n"
    "  ε  = 0.05 AU  (softening)\n"
    "\n"
    "Unit system  (G = 1)\n"
    "  Length   : AU\n"
    "  Mass     : M☉\n"
    "  Time     : yr\n"
    "  Velocity : AU/yr\n"
    "  Energy   : M☉·AU²/yr²\n"
    "\n"
    "Initial state\n"
    f"  m₁ = {MASSES[0]:.3f} M☉\n"
    f"  m₂ = {MASSES[1]:.3f} M☉\n"
    f"  m₃ = {MASSES[2]:.3f} M☉\n"
    f"  E₀ = {E0:.4f} M☉·AU²/yr²\n"
    f"  seed = {SEED}"
)

ax_eq.text(
    0.06, 0.97, EQ_TEXT,
    transform=ax_eq.transAxes,
    color=TEXT, fontsize=7.8,
    va='top', ha='left',
    fontfamily='monospace',
    linespacing=1.55,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0a1020', edgecolor=GRID, alpha=0.7),
)

# ── 3D plot objects ───────────────────────────────────────────────────────────
trails3d = [ax3d.plot([], [], [], '-', color=c, lw=0.9, alpha=0.5)[0] for c in COLORS]
dots3d   = [ax3d.plot([], [], [], 'o', color=c,
                      markersize=6 + MASSES[i]*2.5, zorder=5)[0]
            for i, c in enumerate(COLORS)]

# Body info labels (name + live position + speed)
body_lbls = [
    ax3d.text2D(0.01 + i*0.33, 0.01,
                f'{NAMES[i]}\n{MASSES[i]:.2f} M☉',
                transform=ax3d.transAxes,
                color=COLORS[i], fontsize=7.5, va='bottom',
                fontfamily='monospace')
    for i in range(3)
]
time_lbl = ax3d.text2D(0.02, 0.96, '', transform=ax3d.transAxes,
                        color=TEXT, fontsize=9, va='top', fontfamily='monospace')

# ── Speed slider ──────────────────────────────────────────────────────────────
ax_slider = fig.add_axes([0.20, 0.035, 0.60, 0.025], facecolor='#0a1020')
slider = Slider(
    ax=ax_slider,
    label='Speed  ×',
    valmin=0.25,
    valmax=8.0,
    valinit=1.0,
    valstep=0.25,
    color='#2a9d8f',
)
slider.label.set_color(TEXT)
slider.label.set_fontsize(8)
slider.valtext.set_color(TEXT)
slider.valtext.set_fontsize(8)

# Speed state: mutable so the slider callback can update it
speed = [1.0]

def on_speed_change(val):
    speed[0] = slider.val

slider.on_changed(on_speed_change)

# ── Figure title ──────────────────────────────────────────────────────────────
fig.suptitle('Three-Body Problem  —  Newtonian Gravity  (3D Chaotic Orbit)',
             color=TEXT, fontsize=12, y=0.97)

# ─── Animation ───────────────────────────────────────────────────────────────
BASE_SKIP  = 2       # minimum steps consumed per frame (at speed × 1)
ROT_SPEED  = 0.04    # deg/frame azimuth rotation

# Frame counter: we advance through the pre-computed trajectory using the
# speed multiplier, then wrap around so the animation loops.
frame_idx = [0]


def init():
    for t in trails3d: t.set_data([], []); t.set_3d_properties([])
    for d in dots3d:   d.set_data([], []); d.set_3d_properties([])
    e_line.set_data([], [])
    ke_line.set_data([], [])
    pe_line.set_data([], [])
    return trails3d + dots3d + [e_line, ke_line, pe_line]


def update(anim_frame):
    # Advance the trajectory index by speed-adjusted steps
    advance = max(1, int(BASE_SKIP * speed[0]))
    frame_idx[0] = (frame_idx[0] + advance) % N_STEPS
    f = frame_idx[0]

    # ── 3D orbits ──
    start = max(0, f - TRAIL_LEN)
    for i in range(3):
        trails3d[i].set_data(trajectory[start:f+1, i, 0],
                              trajectory[start:f+1, i, 1])
        trails3d[i].set_3d_properties(trajectory[start:f+1, i, 2])

        pos = trajectory[f, i]
        dots3d[i].set_data([pos[0]], [pos[1]])
        dots3d[i].set_3d_properties([pos[2]])

        # Live position + speed readout in each label
        vel  = velocities[f, i]
        spd  = np.linalg.norm(vel)
        body_lbls[i].set_text(
            f'{NAMES[i]}\n'
            f'{MASSES[i]:.2f} M☉\n'
            f'|v| {spd:.2f} AU/yr\n'
            f'({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) AU'
        )

    # Rotate camera
    ax3d.view_init(elev=25, azim=anim_frame * ROT_SPEED % 360)
    time_lbl.set_text(f't = {f * DT:.2f} yr')

    # ── Energy plots ──
    t_sl = time_arr[:f+1]
    e_line.set_data(t_sl,  energy_tot[:f+1])
    ke_line.set_data(t_sl, energy_ke[:f+1])
    pe_line.set_data(t_sl, energy_pe[:f+1])

    if f > 0:
        d = abs(energy_tot[f] - E0) / abs(E0) * 100
        drift_txt.set_text(f'drift {d:.5f}%')

    return trails3d + dots3d + body_lbls + \
           [e_line, ke_line, pe_line, time_lbl, drift_txt]


ani = animation.FuncAnimation(
    fig, update,
    frames=N_STEPS // BASE_SKIP,
    init_func=init,
    interval=20,
    blit=False    # required for 3D rotation
)

plt.show()
