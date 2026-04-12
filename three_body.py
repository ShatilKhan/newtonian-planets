"""
Three-Body Problem Simulation — Newtonian Mechanics
====================================================

Theory:
-------
Newton's Law of Gravitation:
    F = G * m1 * m2 / r^2  (magnitude)
    F_vec = G * m1 * m2 * (r2 - r1) / |r2 - r1|^3  (vector, force on body 1 from body 2)

Equations of motion for body i:
    a_i = sum_{j != i} [ G * m_j * (r_j - r_i) / |r_j - r_i|^3 ]

State vector: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]

Numerical integration: 4th-order Runge-Kutta (RK4)
    k1 = f(t,       y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h,   y + h   * k3)
    y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# ─── Constants ───────────────────────────────────────────────────────────────
G = 1.0          # Gravitational constant (normalized units)
SOFTENING = 0.01 # Softening length to avoid singularity at r → 0
DT = 0.005       # Time step
N_STEPS = 5000   # Total integration steps
TRAIL_LEN = 300  # How many past positions to draw as trail

# ─── Initial Conditions ──────────────────────────────────────────────────────
# Using a classic "figure-8" periodic orbit (Chenciner & Montgomery, 2000)
# All three bodies have equal mass and chase each other in a figure-8 path.
# Initial positions and velocities from the exact solution:

MASSES = np.array([1.0, 1.0, 1.0])

# Figure-8 initial conditions (normalized)
POS_INIT = np.array([
    [-0.97000436,  0.24308753],
    [ 0.97000436, -0.24308753],
    [ 0.0,         0.0       ],
], dtype=float)

VEL_INIT = np.array([
    [ 0.93240737 / 2,  0.86473146 / 2],
    [ 0.93240737 / 2,  0.86473146 / 2],
    [-0.93240737,     -0.86473146     ],
], dtype=float)

# ─── Physics ─────────────────────────────────────────────────────────────────

def accelerations(positions, masses):
    """
    Compute gravitational acceleration on each body.

    For body i:
        a_i = sum_{j != i} [ G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2) ]

    Softening (eps) prevents numerical blow-up when bodies pass close.
    """
    n = len(masses)
    acc = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = positions[j] - positions[i]          # r_j - r_i
            dist_sq = np.dot(diff, diff) + SOFTENING**2  # |r|^2 + eps^2
            dist_cb = dist_sq ** 1.5                      # (|r|^2 + eps^2)^(3/2)
            acc[i] += G * masses[j] * diff / dist_cb
    return acc


def derivatives(state, masses):
    """
    Build the full derivative vector for RK4.

    state layout: [pos_x1, pos_y1, ..., vel_x1, vel_y1, ...]
    d(pos)/dt = vel
    d(vel)/dt = acc
    """
    n = len(masses)
    positions = state[:n].copy()   # shape (n, 2)
    velocities = state[n:].copy()  # shape (n, 2)
    acc = accelerations(positions, masses)
    return np.concatenate([velocities, acc])


def rk4_step(state, masses, dt):
    """Single RK4 integration step."""
    k1 = derivatives(state,              masses)
    k2 = derivatives(state + 0.5*dt*k1, masses)
    k3 = derivatives(state + 0.5*dt*k2, masses)
    k4 = derivatives(state +     dt*k3, masses)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ─── Integrate ───────────────────────────────────────────────────────────────

def simulate(pos_init, vel_init, masses, n_steps, dt):
    """Run the full simulation, store trajectory."""
    n = len(masses)
    # State: positions stacked then velocities, shape (n, 2) each → flatten
    state = np.concatenate([pos_init, vel_init])  # shape (2n, 2)

    trajectory = np.zeros((n_steps, n, 2))
    trajectory[0] = pos_init.copy()

    for step in range(1, n_steps):
        state = rk4_step(state, masses, dt)
        trajectory[step] = state[:n]

    return trajectory


print("Integrating equations of motion with RK4...")
trajectory = simulate(POS_INIT, VEL_INIT, MASSES, N_STEPS, DT)
print(f"Done. Trajectory shape: {trajectory.shape}  ({N_STEPS} steps × 3 bodies × 2 coords)")

# ─── Animation ───────────────────────────────────────────────────────────────

COLORS = ['#e63946', '#457b9d', '#2a9d8f']  # red, blue, teal
BODY_NAMES = ['Body 1', 'Body 2', 'Body 3']

fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title('Three-Body Problem — Newtonian Mechanics\n(Figure-8 Periodic Orbit)',
             color='white', fontsize=13, pad=12)
ax.tick_params(colors='#555')
for spine in ax.spines.values():
    spine.set_edgecolor('#222')

# Draw grid
ax.grid(True, color='#1e2a38', linewidth=0.5, linestyle='--', alpha=0.6)

# Trail lines
trails = [ax.plot([], [], '-', color=c, linewidth=1.0, alpha=0.5)[0] for c in COLORS]

# Body dots
dots = [ax.plot([], [], 'o', color=c, markersize=10, zorder=5)[0] for c in COLORS]

# Velocity vectors (quivers)
quivers = []

# Labels
labels = [ax.text(0, 0, name, color=c, fontsize=8, ha='center', va='bottom', zorder=6)
          for name, c in zip(BODY_NAMES, COLORS)]

# Time counter
time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                    color='#aaa', fontsize=9, va='top')

# Energy readout (top-right)
energy_text = ax.text(0.98, 0.96, '', transform=ax.transAxes,
                      color='#aaa', fontsize=8, va='top', ha='right')


def kinetic_energy(velocities, masses):
    return sum(0.5 * m * np.dot(v, v) for m, v in zip(masses, velocities))


def potential_energy(positions, masses):
    pe = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i+1, n):
            diff = positions[j] - positions[i]
            r = np.sqrt(np.dot(diff, diff) + SOFTENING**2)
            pe -= G * masses[i] * masses[j] / r
    return pe


def init():
    for trail in trails:
        trail.set_data([], [])
    for dot in dots:
        dot.set_data([], [])
    for label in labels:
        label.set_position((0, 0))
    time_text.set_text('')
    energy_text.set_text('')
    return trails + dots + labels + [time_text, energy_text]


def update(frame):
    # Trail: last TRAIL_LEN frames
    start = max(0, frame - TRAIL_LEN)
    for i, (trail, dot, label) in enumerate(zip(trails, dots, labels)):
        xs = trajectory[start:frame+1, i, 0]
        ys = trajectory[start:frame+1, i, 1]
        trail.set_data(xs, ys)
        # Fade trail alpha based on age
        trail.set_alpha(0.4)

        pos = trajectory[frame, i]
        dot.set_data([pos[0]], [pos[1]])
        label.set_position((pos[0], pos[1] + 0.07))

    t = frame * DT
    time_text.set_text(f't = {t:.2f}')

    # Energy check (every 10 frames to save compute)
    if frame % 10 == 0:
        pos = trajectory[frame]
        # Approximate velocity from finite difference
        if frame > 0:
            vel = (trajectory[frame] - trajectory[frame-1]) / DT
        else:
            vel = VEL_INIT
        ke = kinetic_energy(vel, MASSES)
        pe = potential_energy(pos, MASSES)
        energy_text.set_text(f'KE={ke:.3f}  PE={pe:.3f}\nE={ke+pe:.3f}')

    return trails + dots + labels + [time_text, energy_text]


ani = animation.FuncAnimation(
    fig, update,
    frames=N_STEPS,
    init_func=init,
    interval=16,    # ~60 fps
    blit=True
)

plt.tight_layout()
plt.show()
