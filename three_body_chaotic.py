"""
Three-Body Problem — Chaotic Initial Conditions + Energy Conservation Plot
==========================================================================

Layout: animation (left) | energy plot (right)

Physics:
    Newton's gravitational acceleration on body i:
        a_i = sum_{j != i} [ G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2) ]

    Total energy (conserved quantity):
        E = KE + PE
        KE = sum_i  (1/2) * m_i * |v_i|^2
        PE = sum_{i<j}  -G * m_i * m_j / |r_i - r_j|

    For a correct integrator, E(t) should stay close to E(0).
    Drift in E reveals accumulated numerical error.

Integrator: 4th-order Runge-Kutta (RK4)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# ─── Simulation parameters ───────────────────────────────────────────────────
G          = 1.0
SOFTENING  = 0.05   # larger softening keeps energy well-behaved at close encounters
DT         = 0.002  # smaller step for accuracy
N_STEPS    = 10000
TRAIL_LEN  = 500
SEED       = 42          # change for a different chaotic trajectory

# ─── Random chaotic initial conditions ───────────────────────────────────────
# Strategy: start bodies near vertices of a slightly perturbed equilateral
# triangle so initial separations are never tiny.  Add random velocity
# perturbations to break the symmetry and trigger chaos.
rng = np.random.default_rng(SEED)

MASSES = rng.uniform(0.8, 1.5, size=3)

# Equilateral triangle at radius R, each vertex perturbed randomly
R = 1.0
angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
POS_INIT = np.column_stack([
    R * np.cos(angles) + rng.uniform(-0.15, 0.15, 3),
    R * np.sin(angles) + rng.uniform(-0.15, 0.15, 3),
])

# Circular-orbit tangential velocity scale: v ~ sqrt(G * M_total / R)
v_scale = np.sqrt(G * MASSES.sum() / R) * 0.4
VEL_INIT = rng.uniform(-v_scale, v_scale, size=(3, 2))

# Zero the centre-of-mass velocity so the system stays centred on screen
VEL_INIT -= (MASSES[:, None] * VEL_INIT).sum(axis=0) / MASSES.sum()

print(f"Masses : {MASSES.round(3)}")
print(f"Seed   : {SEED}")

# ─── Physics ─────────────────────────────────────────────────────────────────

def accelerations(pos, masses):
    n = len(masses)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[j] - pos[i]
            r2 = np.dot(d, d) + SOFTENING**2
            acc[i] += G * masses[j] * d / r2**1.5
    return acc


def derivatives(state, masses):
    n = len(masses)
    pos = state[:n]
    vel = state[n:]
    return np.concatenate([vel, accelerations(pos, masses)])


def rk4(state, masses, dt):
    k1 = derivatives(state,              masses)
    k2 = derivatives(state + 0.5*dt*k1,  masses)
    k3 = derivatives(state + 0.5*dt*k2,  masses)
    k4 = derivatives(state +     dt*k3,  masses)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def total_energy(pos, vel, masses):
    n = len(masses)
    ke = sum(0.5 * masses[i] * np.dot(vel[i], vel[i]) for i in range(n))
    pe = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = pos[j] - pos[i]
            r = np.sqrt(np.dot(d, d) + SOFTENING**2)
            pe -= G * masses[i] * masses[j] / r
    return ke + pe, ke, pe


# ─── Integrate ───────────────────────────────────────────────────────────────

print("Integrating...")
state = np.concatenate([POS_INIT, VEL_INIT])

trajectory  = np.zeros((N_STEPS, 3, 2))   # positions
velocities  = np.zeros((N_STEPS, 3, 2))   # velocities
energy_tot  = np.zeros(N_STEPS)
energy_ke   = np.zeros(N_STEPS)
energy_pe   = np.zeros(N_STEPS)
time_arr    = np.arange(N_STEPS) * DT

trajectory[0] = POS_INIT
velocities[0] = VEL_INIT
energy_tot[0], energy_ke[0], energy_pe[0] = total_energy(POS_INIT, VEL_INIT, MASSES)

for step in range(1, N_STEPS):
    state = rk4(state, MASSES, DT)
    trajectory[step] = state[:3]
    velocities[step] = state[3:]
    e, ke, pe = total_energy(state[:3], state[3:], MASSES)
    energy_tot[step] = e
    energy_ke[step]  = ke
    energy_pe[step]  = pe

E0 = energy_tot[0]
print(f"Done.  E₀ = {E0:.4f}   E_final = {energy_tot[-1]:.4f}   "
      f"drift = {abs(energy_tot[-1]-E0)/abs(E0)*100:.4f}%")

# ─── Dynamic axis limits ──────────────────────────────────────────────────────
pad = 0.4
x_all = trajectory[:, :, 0]
y_all = trajectory[:, :, 1]
xlim = (x_all.min() - pad, x_all.max() + pad)
ylim = (y_all.min() - pad, y_all.max() + pad)

# ─── Figure layout ───────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
PANEL_BG  = '#111820'
GRID_COL  = '#1e2a38'
TEXT_COL  = '#c9d1d9'
COLORS    = ['#e63946', '#457b9d', '#2a9d8f']
BODY_NAMES = ['Body 1', 'Body 2', 'Body 3']

fig = plt.figure(figsize=(14, 7), facecolor=DARK_BG)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        left=0.06, right=0.97,
                        top=0.91, bottom=0.09,
                        hspace=0.45, wspace=0.35)

ax_sim   = fig.add_subplot(gs[:, 0])   # left: orbit animation (full height)
ax_etot  = fig.add_subplot(gs[0, 1])   # top-right: total energy
ax_kepe  = fig.add_subplot(gs[1, 1])   # bottom-right: KE and PE

for ax in (ax_sim, ax_etot, ax_kepe):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--', alpha=0.7)

fig.suptitle('Three-Body Problem — Chaotic Orbit + Energy Conservation',
             color=TEXT_COL, fontsize=13, y=0.97)

# ── Orbit panel ──────────────────────────────────────────────────────────────
ax_sim.set_xlim(xlim)
ax_sim.set_ylim(ylim)
ax_sim.set_aspect('equal')
ax_sim.set_title('Orbital Trajectories', color=TEXT_COL, fontsize=10, pad=6)
ax_sim.set_xlabel('x', color=TEXT_COL, fontsize=9)
ax_sim.set_ylabel('y', color=TEXT_COL, fontsize=9)

trails   = [ax_sim.plot([], [], '-',  color=c, lw=0.9, alpha=0.5)[0] for c in COLORS]
dots     = [ax_sim.plot([], [], 'o',  color=c, markersize=8+MASSES[i]*2, zorder=5)[0]
            for i, c in enumerate(COLORS)]
body_lbl = [ax_sim.text(0, 0, f'{n}\nm={MASSES[i]:.2f}',
                         color=c, fontsize=7, ha='center', va='bottom', zorder=6)
            for i, (n, c) in enumerate(zip(BODY_NAMES, COLORS))]
time_txt = ax_sim.text(0.02, 0.97, '', transform=ax_sim.transAxes,
                       color=TEXT_COL, fontsize=9, va='top')

# ── Total energy panel ────────────────────────────────────────────────────────
ax_etot.set_title('Total Energy  E = KE + PE', color=TEXT_COL, fontsize=10, pad=6)
ax_etot.set_xlabel('time', color=TEXT_COL, fontsize=8)
ax_etot.set_ylabel('E', color=TEXT_COL, fontsize=8)

# Draw the reference E₀ line
ax_etot.axhline(E0, color='#888', lw=0.8, linestyle=':', label=f'E₀ = {E0:.3f}')
ax_etot.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)

e_line, = ax_etot.plot([], [], color='#f4a261', lw=1.2)

e_margin = max(abs(energy_tot.max() - E0), abs(energy_tot.min() - E0)) * 1.5 + 1e-6
ax_etot.set_xlim(0, N_STEPS * DT)
ax_etot.set_ylim(E0 - e_margin, E0 + e_margin)

drift_txt = ax_etot.text(0.98, 0.05, '', transform=ax_etot.transAxes,
                          color='#f4a261', fontsize=7, ha='right', va='bottom')

# ── KE / PE panel ─────────────────────────────────────────────────────────────
ax_kepe.set_title('Kinetic & Potential Energy', color=TEXT_COL, fontsize=10, pad=6)
ax_kepe.set_xlabel('time', color=TEXT_COL, fontsize=8)
ax_kepe.set_ylabel('energy', color=TEXT_COL, fontsize=8)

ke_line, = ax_kepe.plot([], [], color='#e63946', lw=1.0, label='KE')
pe_line, = ax_kepe.plot([], [], color='#457b9d', lw=1.0, label='PE')
ax_kepe.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
ax_kepe.set_xlim(0, N_STEPS * DT)
kepe_max = max(energy_ke.max(), abs(energy_pe.min())) * 1.15
ax_kepe.set_ylim(energy_pe.min() * 1.1, kepe_max)

# ─── Animation ───────────────────────────────────────────────────────────────

SKIP = 2   # render every Nth frame for speed (animation shows every SKIP steps)

def init():
    for t in trails: t.set_data([], [])
    for d in dots:   d.set_data([], [])
    e_line.set_data([], [])
    ke_line.set_data([], [])
    pe_line.set_data([], [])
    time_txt.set_text('')
    drift_txt.set_text('')
    return trails + dots + [e_line, ke_line, pe_line, time_txt, drift_txt]


def update(frame):
    f = frame * SKIP   # actual simulation index

    # ── Orbits ──
    start = max(0, f - TRAIL_LEN)
    for i in range(3):
        trails[i].set_data(trajectory[start:f+1, i, 0],
                           trajectory[start:f+1, i, 1])
        pos = trajectory[f, i]
        dots[i].set_data([pos[0]], [pos[1]])
        body_lbl[i].set_position((pos[0], pos[1] + 0.08))

    time_txt.set_text(f't = {f * DT:.2f}')

    # ── Energy plots ──
    t_slice = time_arr[:f+1]
    e_line.set_data(t_slice,  energy_tot[:f+1])
    ke_line.set_data(t_slice, energy_ke[:f+1])
    pe_line.set_data(t_slice, energy_pe[:f+1])

    # Energy drift label
    if f > 0:
        drift = abs(energy_tot[f] - E0) / abs(E0) * 100
        drift_txt.set_text(f'drift {drift:.4f}%')

    return trails + dots + body_lbl + [e_line, ke_line, pe_line, time_txt, drift_txt]


n_frames = N_STEPS // SKIP

ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=init,
    interval=20,
    blit=True
)

plt.show()
