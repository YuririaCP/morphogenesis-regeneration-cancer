"""
sim2d_phasesweep.py
Yuriria Cortés-Poza
===================
2D Phase sweep: coherence as a function of (G_bar, alpha).

For each parameter point (G_bar, alpha) in a 6x6 grid:
  - Fix gap-junction conductances at G_ij = G_bar for all pairs.
  - Fix memory at m_i = IDENT (committed binary pattern).
  - Integrate only the V subsystem from a random IC for T=150.
  - Measure coherence C = fraction of cells where V_i and phi(m_i)
    have the same sign (i.e. correct half-plane).

This is the 2D analogue of the 1D phase diagram in prove_threshold.py,
validating Corollary 1 (G_bar + alpha > alpha_c* => monostability) in 2D.

The single-cell reference threshold  G_bar + alpha = alpha_c* = 0.5625
is overlaid as a dashed line.

Output: figs/fig_2d_phasesweep.pdf
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

# ─── Model parameters ────────────────────────────────────────────────────────
N = 20; NC = N * N
a = 1.0; L = 1.5; Vp = L; Vm = -L
C = 1.0; k_sig = 4.0
alpha_c_star = a * L**2 / 4   # 0.5625
T_SWEEP = 150.0

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

IDENT = np.array([1.0 if c < N//2 else 0.0
                  for r in range(N) for c in range(N)])

def phi(m):  return Vm + (Vp - Vm) * m
PHI_I = phi(IDENT)

def f_bist(V): return a * V * (V + L) * (V - L)

def lap_V_fixed(V, G_bar):
    """Laplacian with fixed uniform conductance G_bar."""
    lv = np.zeros(NC); d = V[PJ] - V[PI]
    np.add.at(lv, PI,  G_bar * d)
    np.add.at(lv, PJ, -G_bar * d)
    return lv

def make_rhs_Vonly(alpha, G_bar):
    """V-only RHS with fixed G and m=IDENT."""
    def rhs(t, V):
        return (1/C) * (-f_bist(V) + alpha*(PHI_I - V) + lap_V_fixed(V, G_bar))
    return rhs

def coherence(V):
    """Fraction of cells where V_i and phi_i have the same sign."""
    return np.mean((V > 0) == (PHI_I > 0))

# ─── Parameter grid ──────────────────────────────────────────────────────────
G_vals     = np.array([0.02, 0.10, 0.20, 0.35, 0.55, 0.80])
alpha_vals = np.array([0.05, 0.15, 0.30, 0.45, 0.65, 0.85])
NG, NA = len(G_vals), len(alpha_vals)

COH = np.zeros((NA, NG))

print(f'Phase sweep: {NA}x{NG} = {NA*NG} simulations, T={T_SWEEP}')
print(f'alpha_c* = {alpha_c_star:.4f}')

rng = np.random.default_rng(99)
seeds = rng.integers(0, 10000, size=(NA, NG))

for i, alpha in enumerate(alpha_vals):
    for j, G_bar in enumerate(G_vals):
        V0 = np.random.default_rng(int(seeds[i,j])).uniform(Vm, Vp, NC)
        t_eval = np.linspace(0, T_SWEEP, 80)
        sol = solve_ivp(make_rhs_Vonly(alpha, G_bar),
                        (0, T_SWEEP), V0, method='RK45',
                        t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=1.0)
        COH[i, j] = coherence(sol.y[:, -1])
        print(f'  (G={G_bar:.2f}, alpha={alpha:.2f}): '
              f'G+alpha={G_bar+alpha:.3f}, C={COH[i,j]:.3f}')

print(f'\nCoherence range: [{COH.min():.3f}, {COH.max():.3f}]')

# ─── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 4.5))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0], wspace=0.38,
                          left=0.08, right=0.97, bottom=0.14, top=0.88)

# ── Left: heatmap ─────────────────────────────────────────────────────────────
ax = fig.add_subplot(outer[0])
cmap_sweep = plt.cm.RdYlGn
im = ax.pcolormesh(G_vals, alpha_vals, COH,
                   cmap=cmap_sweep, vmin=0.5, vmax=1.0, shading='auto')
cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
cb.set_label(r'Coherence $\mathcal{C}$', fontsize=9)
cb.ax.tick_params(labelsize=7)

# Analytic phase boundary G + alpha = alpha_c*
G_line = np.linspace(0, alpha_c_star, 200)
ax.plot(G_line, alpha_c_star - G_line, 'k--', lw=2.0,
        label=r'$\bar{G}+\alpha = \alpha_c^*$  (single-cell reference threshold)')

# Contour at C=0.80 from simulation
try:
    cs = ax.contour(G_vals, alpha_vals, COH, levels=[0.80],
                    colors=['navy'], linestyles=['-.'], linewidths=[1.5])
    ax.clabel(cs, fmt=r'$\mathcal{C}=0.80$', fontsize=7.5)
except Exception:
    pass

ax.set_xlabel(r'Fixed coupling $\bar{G}$ (per-pair)', fontsize=10)
ax.set_ylabel(r'Error-correction strength $\alpha$', fontsize=10)
ax.set_title(r'2D coherence phase diagram', fontsize=10, pad=5)
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.88)
ax.tick_params(labelsize=8)
ax.text(alpha_c_star*0.12, alpha_c_star*0.08,
        r'Fragmented / target disagreement', fontsize=8, color='white', va='bottom',
        bbox=dict(boxstyle='round,pad=0.18', facecolor='#8b0000',
                  edgecolor='none', alpha=0.85))
ax.text(alpha_c_star*0.10, alpha_vals[-1]*0.92,
        r'Morphostasis', fontsize=8, color='darkgreen', va='top')

# ── Right: representative V fields at 4 corners ───────────────────────────────
corners = [
    (0, 0,  'Low $\\bar G$, low $\\alpha$\n(fragmented)'),
    (0, -1, 'High $\\bar G$, low $\\alpha$\n(morphostatic)'),
    (-1, 0, 'Low $\\bar G$, high $\\alpha$\n(morphostatic)'),
    (-1,-1, 'High $\\bar G$, high $\\alpha$\n(morphostatic)'),
]

inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1],
                                         hspace=0.60, wspace=0.40)

for idx, (ai, gj, label) in enumerate(corners):
    alpha_c = alpha_vals[ai]; G_c = G_vals[gj]
    V0 = np.random.default_rng(777 + idx).uniform(Vm, Vp, NC)
    t_eval = np.linspace(0, T_SWEEP, 60)
    sol_c = solve_ivp(make_rhs_Vonly(alpha_c, G_c),
                      (0, T_SWEEP), V0, method='RK45',
                      t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=1.0)
    V_fin = sol_c.y[:, -1].reshape(N, N)
    r_idx = idx // 2; c_idx = idx % 2
    ax_c = fig.add_subplot(inner[r_idx, c_idx])
    ax_c.imshow(V_fin, cmap='RdBu_r', vmin=Vm, vmax=Vp,
                origin='upper', interpolation='nearest')
    ax_c.set_xticks([]); ax_c.set_yticks([])
    coh_c = coherence(sol_c.y[:, -1])
    ax_c.set_title(f'{label}\n$\\mathcal{{C}}={coh_c:.2f}$',
                   fontsize=7, pad=2)

fig.suptitle(
    r'2D phase sweep: coherence $\mathcal{C}(\bar G, \alpha)$ vs single-cell reference threshold '
    r'$\bar G + \alpha = \alpha_c^* = 0.5625$',
    fontsize=9.5, y=1.03, style='italic', color='#333333')

outpath = 'figs/fig_2d_phasesweep.pdf'
fig.savefig(outpath, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {outpath}')
print('Done.')
