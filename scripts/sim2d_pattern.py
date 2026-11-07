"""
sim2d_pattern.py
Yuriria Cortés-Poza
===================
2D morphogenesis with a complex (non-stripe) morphological identity.

Disc identity:  cells within radius R=7 of the grid centre target V_+;
                peripheral cells target V_-.

This demonstrates that the model supports arbitrary spatial patterns
imposed by the identity field IDENT, not just stripe morphologies.

The figure shows:
  Row 0 – Identity pattern (target map phi(IDENT))
  Row 1 – Morphogenesis from random IC: snapshots at t=0,100,250,500
  Row 2 – Comparison: stripe attractor (left) vs disc attractor (right)
           with their respective memory fields m_i

Output: figs/fig_2d_pattern.pdf
"""

import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import os

os.makedirs('figs', exist_ok=True)

# ─── Parameters ──────────────────────────────────────────────────────────────
N = 20; NC = N * N
a = 1.0; L = 1.5; Vp = L; Vm = -L
C = 1.0; ALPHA = 1.2
gamma = 2.0; lam_G = 0.5; lam_m = 0.3; kappa = 0.1
G0 = 1.0; tau_G = 5.0; tau_m = 50.0; k_sig = 4.0
T_MORPH = 500.0

# ─── Grid ────────────────────────────────────────────────────────────────────
PAIRS = []
for r in range(N):
    for c in range(N):
        i = r*N + c
        if r+1 < N: PAIRS.append((i, (r+1)*N+c))
        if c+1 < N: PAIRS.append((i, r*N+c+1))
NPAIRS = len(PAIRS)
PI = np.array([p[0] for p in PAIRS], dtype=int)
PJ = np.array([p[1] for p in PAIRS], dtype=int)

def f_bist(V):  return a * V * (V + L) * (V - L)
def sig(V):     return 1.0 / (1.0 + np.exp(-k_sig * V))
def phi(m):     return Vm + (Vp - Vm) * m

def lap_V(V, G):
    lv = np.zeros(NC); d = V[PJ] - V[PI]
    np.add.at(lv, PI, G*d); np.add.at(lv, PJ, -G*d); return lv

def lap_m(m, G):
    lm = np.zeros(NC); d = m[PJ] - m[PI]
    np.add.at(lm, PI, G*d); np.add.at(lm, PJ, -G*d); return lm

# ─── Identity patterns ────────────────────────────────────────────────────────
def stripe_ident():
    return np.array([1.0 if c < N//2 else 0.0
                     for r in range(N) for c in range(N)])

def disc_ident(R=7):
    cx, cy = (N-1)/2, (N-1)/2
    return np.array([1.0 if (r-cx)**2 + (c-cy)**2 <= R**2 else 0.0
                     for r in range(N) for c in range(N)])

def quadrant_ident():
    """Checkerboard: top-left & bottom-right = 1, others = 0."""
    return np.array([
        1.0 if (r < N//2 and c < N//2) or (r >= N//2 and c >= N//2) else 0.0
        for r in range(N) for c in range(N)])

# Use disc for experiment, also compute stripe for comparison
IDENT_DISC    = disc_ident(R=7)
IDENT_STRIPE  = stripe_ident()
IDENT_QUAD    = quadrant_ident()

# ─── ODE factory ─────────────────────────────────────────────────────────────
def make_rhs(ident):
    phi_i = phi(ident)
    def rhs(t, y):
        V = y[:NC]; Gp = y[NC:NC+NPAIRS]; m = y[NC+NPAIRS:]
        dV = (1/C)*(-f_bist(V) + ALPHA*(phi_i - V) + lap_V(V, Gp))
        dG = (1/tau_G)*(sig(V[PI])*sig(V[PJ])*(1+gamma*m[PI]*m[PJ]) - lam_G*Gp)
        dm = (1/tau_m)*(lam_m*(sig(V) - m) + kappa*lap_m(m, Gp))
        return np.concatenate([dV, dG, dm])
    return rhs

def run_morph(ident, seed, T=T_MORPH, n_eval=300):
    rng = np.random.default_rng(seed)
    V  = rng.uniform(Vm, Vp, NC)
    G  = rng.uniform(0.0, 0.2, NPAIRS)
    m  = rng.uniform(0.0, 1.0, NC)
    y0 = np.concatenate([V, G, m])
    t_eval = np.linspace(0, T, n_eval)
    sol = solve_ivp(make_rhs(ident), (0, T), y0, method='RK45',
                    t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=0.5)
    return sol, t_eval

# ─── Run simulations ─────────────────────────────────────────────────────────
print('Disc morphogenesis ...')
sol_disc, t_disc   = run_morph(IDENT_DISC, seed=7)
print('Stripe morphogenesis ...')
sol_stripe, t_str  = run_morph(IDENT_STRIPE, seed=7)
print('Quadrant morphogenesis ...')
sol_quad, t_quad   = run_morph(IDENT_QUAD, seed=7)

def extract(sol, t_arr, t_target):
    idx = np.argmin(np.abs(t_arr - t_target))
    y = sol.y[:, idx]
    return y[:NC].reshape(N,N), y[NC+NPAIRS:].reshape(N,N)

SNAP_T = [0, 100, 250, 500]
CMAP_V = 'RdBu_r'; CMAP_M = 'viridis'

# ─── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 5, hspace=0.50, wspace=0.22,
                        left=0.07, right=0.97, top=0.94, bottom=0.05)

def show_V(ax, V, title='', cbar=False):
    im = ax.imshow(V, cmap=CMAP_V, vmin=Vm, vmax=Vp,
                   origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=8.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    if cbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
        cb.ax.tick_params(labelsize=7)
    return im

def show_m(ax, m, title='', cbar=False):
    im = ax.imshow(m, cmap=CMAP_M, vmin=0, vmax=1,
                   origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=8.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    if cbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
        cb.ax.tick_params(labelsize=7)
    return im

# ── Row 0: identity patterns (targets) ───────────────────────────────────────
for col, (ident, label) in enumerate([
        (IDENT_STRIPE, 'Stripe\n(left $V_+$, right $V_-$)'),
        (IDENT_DISC,   'Disc\n(centre $V_+$, edge $V_-$)'),
        (IDENT_QUAD,   'Checkerboard\n($V_+$ diagonal)'),
]):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(phi(ident).reshape(N,N), cmap=CMAP_V, vmin=Vm, vmax=Vp,
                   origin='upper', interpolation='nearest')
    ax.set_title(label, fontsize=8.5, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel('Identity\n' + r'$\phi(\mathrm{IDENT})$', fontsize=8.5)
    if col == 2:
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04,
                     label=r'$\phi$').ax.tick_params(labelsize=7)

fig.add_subplot(gs[0, 3]).axis('off')   # placeholder
fig.add_subplot(gs[0, 4]).axis('off')

# ── Row 1: Disc morphogenesis snapshots ──────────────────────────────────────
for col, t_snap in enumerate(SNAP_T):
    V_snap, _ = extract(sol_disc, t_disc, t_snap)
    ax = fig.add_subplot(gs[1, col])
    show_V(ax, V_snap, f'$t={t_snap}$', cbar=(col==3))
    if col == 0:
        ax.set_ylabel('Disc morphogenesis\n$V_i$', fontsize=8.5)

# memory at convergence
_, m_disc = extract(sol_disc, t_disc, 500)
ax = fig.add_subplot(gs[1, 4])
show_m(ax, m_disc, '$t=500$, memory $m_i$', cbar=True)

# ── Row 2: Quadrant morphogenesis snapshots ───────────────────────────────────
for col, t_snap in enumerate(SNAP_T):
    V_snap, _ = extract(sol_quad, t_quad, t_snap)
    ax = fig.add_subplot(gs[2, col])
    show_V(ax, V_snap, f'$t={t_snap}$', cbar=(col==3))
    if col == 0:
        ax.set_ylabel('Checkerboard morphogenesis\n$V_i$', fontsize=8.5)

_, m_quad = extract(sol_quad, t_quad, 500)
ax = fig.add_subplot(gs[2, 4])
show_m(ax, m_quad, '$t=500$, memory $m_i$', cbar=True)

# ── Coherence stats ───────────────────────────────────────────────────────────
for name, sol, t_arr, ident in [
        ('Disc',       sol_disc, t_disc, IDENT_DISC),
        ('Stripe',     sol_stripe, t_str, IDENT_STRIPE),
        ('Checkerboard', sol_quad, t_quad, IDENT_QUAD)]:
    V_fin = sol.y[:NC, -1]
    phi_fin = phi(ident)
    coh = np.sum((V_fin > 0) == (phi_fin > 0)) / NC
    print(f'  {name}: coherence = {coh:.3f}')

fig.suptitle(
    r'2D morphogenesis with complex identity patterns  '
    r'($\alpha=1.2>\alpha_c^*$, random initial condition)',
    fontsize=10, y=0.997, style='italic', color='#333333')

outpath = 'figs/fig_2d_pattern.pdf'
fig.savefig(outpath, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {outpath}')
print('Done.')
