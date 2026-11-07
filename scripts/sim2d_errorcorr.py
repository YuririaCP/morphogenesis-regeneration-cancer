"""
sim2d_errorcorr.py
Yuriria Cortés-Poza
===================
2D Active Error Correction after structured mispatterning.

Design
------
Phase 1 : Establish the 2D morphostatic attractor (alpha=1.2, T=250).
Phase 2 : In Omega = 6x6 patch in the depolarised (left) half:
            - Zero gap-junction conductances touching Omega.
            - Invert V_i -> -V_i  (memory m_i unchanged).
Phase 3 : Evolve with modified alpha; track E_Omega(t).

Case A  alpha=1.2 > alpha_c*=0.5625 : error-correction dominates; V restored.
Case B  alpha=0.10 < alpha_c*       : no restoring force; inverted state persists.

Output: figs/fig_2d_errorcorr.pdf
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

# ─── Model parameters (same as sim2d.py) ────────────────────────────────────
N  = 20; NC = N * N
a  = 1.0; L = 1.5; Vp = L; Vm = -L
alpha_c_star = a * L**2 / 4          # 0.5625
C = 1.0
gamma = 2.0; lam_G = 0.5; lam_m = 0.3; kappa = 0.1
G0_init = 1.0; tau_G = 5.0; tau_m = 50.0; k_sig = 4.0

ALPHA_A = 1.2    # morphostatic (corrects)
ALPHA_B = 0.10   # sub-threshold (fails)
T_PRE   = 250.0
T_POST  = 250.0

# ─── Grid helpers ────────────────────────────────────────────────────────────
PAIRS = []
for r in range(N):
    for c in range(N):
        i = r * N + c
        if r + 1 < N: PAIRS.append((i, (r+1)*N + c))
        if c + 1 < N: PAIRS.append((i, r*N + c + 1))
NPAIRS = len(PAIRS)
PI = np.array([p[0] for p in PAIRS], dtype=int)
PJ = np.array([p[1] for p in PAIRS], dtype=int)

IDENT = np.array([1.0 if c < N//2 else 0.0
                  for r in range(N) for c in range(N)])

def f_bist(V):  return a * V * (V + L) * (V - L)
def sig(V):     return 1.0 / (1.0 + np.exp(-k_sig * V))
def phi(m):     return Vm + (Vp - Vm) * m

def lap_V(V, G):
    lv = np.zeros(NC); d = V[PJ] - V[PI]
    np.add.at(lv, PI,  G * d); np.add.at(lv, PJ, -G * d); return lv

def lap_m(m, G):
    lm = np.zeros(NC); d = m[PJ] - m[PI]
    np.add.at(lm, PI,  G * d); np.add.at(lm, PJ, -G * d); return lm

def make_rhs(alpha):
    phi_i = phi(IDENT)
    def rhs(t, y):
        V  = y[:NC]; Gp = y[NC:NC+NPAIRS]; m = y[NC+NPAIRS:]
        dV = (1/C) * (-f_bist(V) + alpha*(phi_i - V) + lap_V(V, Gp))
        dG = (1/tau_G) * (sig(V[PI])*sig(V[PJ])*(1 + gamma*m[PI]*m[PJ]) - lam_G*Gp)
        dm = (1/tau_m) * (lam_m*(sig(V) - m) + kappa*lap_m(m, Gp))
        return np.concatenate([dV, dG, dm])
    return rhs

def run_ivp(y0, T, alpha, n_eval=300):
    t_eval = np.linspace(0, T, n_eval)
    sol = solve_ivp(make_rhs(alpha), (0, T), y0, method='RK45',
                    t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=0.5)
    return sol, t_eval

# ─── Phase 1: establish morphostatic attractor ───────────────────────────────
print('=== Phase 1: establishing 2D attractor ===')
rng = np.random.default_rng(0)
V_ic = phi(IDENT) + rng.normal(0, 0.05, NC)
G_ic = G0_init * np.ones(NPAIRS) + rng.normal(0, 0.02, NPAIRS)
G_ic = np.clip(G_ic, 0, None)
m_ic = IDENT.copy() + rng.normal(0, 0.02, NC)
m_ic = np.clip(m_ic, 0, 1)
y_pre = np.concatenate([V_ic, G_ic, m_ic])

sol_pre, _ = run_ivp(y_pre, T_PRE, ALPHA_A, n_eval=100)
y_star = sol_pre.y[:, -1]
V_star = y_star[:NC]
print(f'  Attractor V range: [{V_star.min():.3f}, {V_star.max():.3f}]')

# ─── Omega: 6x6 patch in left (depolarised) half ─────────────────────────────
OMEGA = np.array([r*N + c for r in range(7, 13) for c in range(2, 8)])
OMEGA_SET = set(OMEGA.tolist())
OMEGA_PAIRS = np.array([k for k,(pi,pj) in enumerate(PAIRS)
                         if pi in OMEGA_SET or pj in OMEGA_SET], dtype=int)
print(f'  Omega: {len(OMEGA)} cells, {len(OMEGA_PAIRS)} touching pairs')

phi_omega = phi_vals = phi(IDENT[OMEGA])
print(f'  phi(m) in Omega: mean={phi_omega.mean():.3f}  (target V+={Vp})')
print(f'  V_star in Omega: mean={V_star[OMEGA].mean():.3f}')

# ─── Apply perturbation ───────────────────────────────────────────────────────
y_pert = y_star.copy()
y_pert[OMEGA] = -y_pert[OMEGA]                  # invert V in Omega
y_pert[OMEGA] = np.clip(y_pert[OMEGA], Vm, Vp)
y_pert[NC + OMEGA_PAIRS] = 0.0                  # zero G in Omega

E0 = np.mean(np.abs(y_pert[OMEGA] - V_star[OMEGA]))
print(f'  Initial error E_Omega = {E0:.3f}')

# ─── Phase 3: two alpha cases ─────────────────────────────────────────────────
print(f'\n=== Case A: alpha={ALPHA_A} (corrects) ===')
solA, t_post = run_ivp(y_pert.copy(), T_POST, ALPHA_A, n_eval=300)
print(f'=== Case B: alpha={ALPHA_B} (fails) ===')
solB, _      = run_ivp(y_pert.copy(), T_POST, ALPHA_B, n_eval=300)

E_A = np.array([np.mean(np.abs(solA.y[OMEGA, k] - V_star[OMEGA])) for k in range(300)])
E_B = np.array([np.mean(np.abs(solB.y[OMEGA, k] - V_star[OMEGA])) for k in range(300)])
print(f'  E_A: {E_A[0]:.3f} → {E_A[-1]:.3f}')
print(f'  E_B: {E_B[0]:.3f} → {E_B[-1]:.3f}')

# ─── Figure ──────────────────────────────────────────────────────────────────
SNAP_T   = [0, 50, 150, 250]
SNAP_IDX = [np.argmin(np.abs(t_post - s)) for s in SNAP_T]
CMAP_V   = 'RdBu_r'

omega_mask = np.zeros((N, N))
for idx in OMEGA:
    r, c = divmod(int(idx), N); omega_mask[r, c] = 1.0

fig = plt.figure(figsize=(12.5, 6.8))
gs  = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.88],
                        hspace=0.50, wspace=0.16,
                        left=0.07, right=0.94, top=0.92, bottom=0.09)

def show_snap(ax, y_vec, t_label, ylabel=None):
    V  = y_vec[:NC].reshape(N, N)
    im = ax.imshow(V, cmap=CMAP_V, vmin=Vm, vmax=Vp,
                   origin='upper', interpolation='nearest')
    ax.contour(omega_mask, levels=[0.5], colors='gold', linewidths=1.1)
    ax.set_title(f'$t = {t_label}$', fontsize=8.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    return im

for col, (si, st) in enumerate(zip(SNAP_IDX, SNAP_T)):
    ax = fig.add_subplot(gs[0, col])
    lbl = rf'(a)  $\alpha={ALPHA_A}$ — corrects' if col == 0 else None
    show_snap(ax, solA.y[:, si], st, lbl)

for col, (si, st) in enumerate(zip(SNAP_IDX, SNAP_T)):
    ax = fig.add_subplot(gs[1, col])
    lbl = rf'(b)  $\alpha={ALPHA_B}$ — trapped' if col == 0 else None
    show_snap(ax, solB.y[:, si], st, lbl)

# shared colourbar
cax = fig.add_axes([0.955, 0.42, 0.013, 0.47])
sm  = plt.cm.ScalarMappable(cmap=CMAP_V, norm=plt.Normalize(Vm, Vp))
cb  = fig.colorbar(sm, cax=cax); cb.set_label(r'$V_i$', fontsize=9)
cb.ax.tick_params(labelsize=7)

# E_Omega panel
ax_E = fig.add_subplot(gs[2, :])
ax_E.plot(t_post, E_A, color='#1a6faf', lw=2.2,
          label=(rf'$\alpha={ALPHA_A}$  — correction succeeds '
                 r'($E_\Omega \to 0$)'))
ax_E.plot(t_post, E_B, color='#cc2222', lw=2.2, ls='--',
          label=(rf'$\alpha={ALPHA_B}$  — trapped in inverted state '
                 r'($E_\Omega$ persists)'))
ax_E.axhline(0, color='gray', lw=0.7, ls=':')
ax_E.set_xlabel('Time since perturbation', fontsize=10)
ax_E.set_ylabel(
    r'$E_\Omega(t)=\frac{1}{|\Omega|}\sum_{i\in\Omega}|V_i-V_i^*|$',
    fontsize=9.5)
ax_E.set_title(r'Error relative to morphostatic reference $V^*$',
               fontsize=9.5, pad=3)
ax_E.legend(fontsize=8.5, loc='center right', framealpha=0.88)
ax_E.set_xlim(0, T_POST); ax_E.set_ylim(bottom=0)
ax_E.spines['top'].set_visible(False); ax_E.spines['right'].set_visible(False)
ax_E.tick_params(labelsize=8)
ax_E.annotate(
    rf'$\alpha_c^* = {alpha_c_star:.4f}$  (monostability threshold)',
    xy=(0.50, 0.90), xycoords='axes fraction', fontsize=8,
    color='#555555', ha='center',
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
              edgecolor='lightgray', alpha=0.90))

fig.suptitle(
    r'2D active error correction — $\Omega$ = 6$\times$6 patch, '
    r'$G_\Omega \!=\! 0$ and $V_\Omega$ inverted at $t=0$',
    fontsize=9.5, y=0.975, style='italic', color='#333333')

outpath = 'figs/fig_2d_errorcorr.pdf'
fig.savefig(outpath, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {outpath}')
print('Done.')
