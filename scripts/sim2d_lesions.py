"""
sim2d_lesions.py
Yuriria Cortés-Poza
===================
2D recovery from three distinct structured-perturbation geometries.

Three ablation shapes applied to the morphostatic 2D attractor:
  (A) Square  : 6x6 patch crossing the left-right morphological boundary
                (rows 7-12, cols 7-12)
  (B) Band    : full-width horizontal band (rows 9-10, all cols)
  (C) Dual    : two separate 4x4 lesions, one per half
                (left: rows 3-7, cols 3-7 | right: rows 12-16, cols 12-16)

Ablation: scramble V to Uniform(-0.3,0.3), m to Uniform(0,0.5).
Recovery: alpha=1.2 (morphostatic regime, guarantees single morphostatic
          attractor per Theorem 1 + Corollary 1).

Four snapshots per geometry:
  (i)  t=200  pre-lesion (converged attractor)
  (ii) t=200  post-ablation
  (iii)t=350  mid-recovery
  (iv) t=600  fully recovered

Output: figs/fig_2d_lesions.pdf
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
C = 1.0; ALPHA = 1.2
gamma = 2.0; lam_G = 0.5; lam_m = 0.3; kappa = 0.1
G0 = 1.0; tau_G = 5.0; tau_m = 50.0; k_sig = 4.0

T_PRE  = 200.0
T_ABL  = 400.0   # recovery time after ablation

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

def f_bist(V):  return a * V * (V + L) * (V - L)
def sig(V):     return 1.0 / (1.0 + np.exp(-k_sig * V))
def phi(m):     return Vm + (Vp - Vm) * m

def lap_V(V, G):
    lv = np.zeros(NC); d = V[PJ] - V[PI]
    np.add.at(lv, PI, G*d); np.add.at(lv, PJ, -G*d); return lv

def lap_m(m, G):
    lm = np.zeros(NC); d = m[PJ] - m[PI]
    np.add.at(lm, PI, G*d); np.add.at(lm, PJ, -G*d); return lm

PHI_I = phi(IDENT)

def rhs(t, y):
    V = y[:NC]; Gp = y[NC:NC+NPAIRS]; m = y[NC+NPAIRS:]
    dV = (1/C) * (-f_bist(V) + ALPHA*(PHI_I - V) + lap_V(V, Gp))
    dG = (1/tau_G) * (sig(V[PI])*sig(V[PJ])*(1+gamma*m[PI]*m[PJ]) - lam_G*Gp)
    dm = (1/tau_m) * (lam_m*(sig(V) - m) + kappa*lap_m(m, Gp))
    return np.concatenate([dV, dG, dm])

def solve_phase(y0, t_span, n_eval=200):
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(rhs, t_span, y0, method='RK45',
                    t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=0.5)
    return sol

# ─── Morphostatic initial condition ─────────────────────────────────────────
def morphostatic_IC(seed=0):
    rng = np.random.default_rng(seed)
    V = PHI_I + rng.normal(0, 0.05, NC)
    G = G0 * np.ones(NPAIRS) + rng.normal(0, 0.02, NPAIRS)
    G = np.clip(G, 0, None)
    m = IDENT.copy() + rng.normal(0, 0.02, NC)
    m = np.clip(m, 0, 1)
    return np.concatenate([V, G, m])

# ─── Phase 1: establish attractor ────────────────────────────────────────────
print('Phase 1: establishing attractor ...')
y0 = morphostatic_IC(seed=0)
sol_pre = solve_phase(y0, (0, T_PRE), n_eval=60)
y_att = sol_pre.y[:, -1]
print(f'  V range [{y_att[:NC].min():.2f}, {y_att[:NC].max():.2f}]')

# ─── Lesion geometry definitions ──────────────────────────────────────────────
def cells_from_rc(rows, cols):
    return np.array([r*N + c for r in rows for c in cols], dtype=int)

LESIONS = {
    'A_square': cells_from_rc(range(7, 13), range(7, 13)),   # 6x6, crosses midline
    'B_band'  : cells_from_rc(range(9, 11), range(0, N)),    # full-width band
    'C_dual'  : np.concatenate([                              # two 4x4 patches
                    cells_from_rc(range(3, 7), range(3, 7)),
                    cells_from_rc(range(13, 17), range(13, 17))]),
}
LESION_LABELS = {
    'A_square': '(A) Square perturbation: 6×6, crosses midline',
    'B_band'  : '(B) Band perturbation: full-width horizontal band',
    'C_dual'  : '(C) Dual perturbation: two separate 4×4 patches',
}

def apply_ablation(y, lesion_cells, rng):
    """Scramble V and m in lesion; also zero G on touching pairs."""
    yp = y.copy()
    yp[lesion_cells] = rng.uniform(-0.3, 0.3, len(lesion_cells))
    lset = set(lesion_cells.tolist())
    for k, (pi, pj) in enumerate(PAIRS):
        if pi in lset or pj in lset:
            yp[NC + k] = 0.0
    yp[NC+NPAIRS + lesion_cells] = rng.uniform(0.0, 0.5, len(lesion_cells))
    return yp

# ─── Run all three lesion experiments ────────────────────────────────────────
rng_abl = np.random.default_rng(42)

SNAP_TIMES = [200, 200, 350, 600]   # pre-lesion, post-lesion, mid, final
results = {}   # key -> (y_pre, y_post, sol_rec)

for key, lesion_cells in LESIONS.items():
    print(f'\nLesion {key}: {len(lesion_cells)} cells ...')
    y_abl = apply_ablation(y_att, lesion_cells, rng_abl)
    print(f'  E at ablation: {np.mean(np.abs(y_abl[lesion_cells] - y_att[lesion_cells])):.3f}')
    sol_rec = solve_phase(y_abl, (T_PRE, T_PRE + T_ABL), n_eval=250)
    V_final = sol_rec.y[:NC, -1]
    # coherence: fraction of cells in correct half-plane
    correct = np.sum((V_final > 0) == (PHI_I > 0)) / NC
    print(f'  Final coherence: {correct:.3f}')
    results[key] = (y_att, y_abl, sol_rec)

# ─── Figure ──────────────────────────────────────────────────────────────────
CMAP_V = 'RdBu_r'
T_REC = T_PRE + T_ABL   # = 600

def snap_at(sol_rec, t_abs, T_start, T_end):
    """Return y_vec from recovery solution at absolute time t_abs."""
    t_arr = np.linspace(T_start, T_end, sol_rec.y.shape[1])
    idx = np.argmin(np.abs(t_arr - t_abs))
    return sol_rec.y[:, idx]

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(3, 5, hspace=0.48, wspace=0.18,
                        left=0.10, right=0.95, top=0.90, bottom=0.06)

# Row label y-centres computed from GridSpec geometry:
# total height = 0.90-0.06 = 0.84; 3 rows with hspace=0.48
# h = 0.84 / (3 + 2*0.48) ≈ 0.2105; gap = 0.48*h ≈ 0.1010
_h   = (0.90 - 0.06) / (3 + 2*0.48)
_gap = 0.48 * _h
_row_centers = [0.90 - _h*(k+0.5) - _gap*k for k in range(3)]

col_titles = ['Pre-perturbation\n$t=200$',
              'Post-perturbation\n$t=200+\epsilon$',
              'Mid-recovery\n$t=350$',
              'Recovered\n$t=600$',
              'Memory $m_i$\n$t=600$']

for col, ttl in enumerate(col_titles):
    fig.text(0.115 + col*0.171, 0.935, ttl,
             ha='center', fontsize=8.5, va='top', color='#333333')

for row, (key, lesion_cells) in enumerate(LESIONS.items()):
    y_att_snap, y_abl, sol_rec = results[key]
    snaps_V = [
        y_att_snap,
        y_abl,
        snap_at(sol_rec, 350, T_PRE, T_PRE+T_ABL),
        snap_at(sol_rec, 600, T_PRE, T_PRE+T_ABL),
    ]

    # row label via fig.text so it never reaches the column-title row
    fig.text(0.005, _row_centers[row], LESION_LABELS[key],
             rotation=90, ha='center', va='center', fontsize=8.2,
             color='#333333')

    # lesion mask for contour overlay
    lmask = np.zeros((N, N))
    for idx in lesion_cells:
        r, c = divmod(int(idx), N); lmask[r, c] = 1.0

    for col in range(4):
        ax = fig.add_subplot(gs[row, col])
        V = snaps_V[col][:NC].reshape(N, N)
        im = ax.imshow(V, cmap=CMAP_V, vmin=Vm, vmax=Vp,
                       origin='upper', interpolation='nearest')
        if col <= 2:
            ax.contour(lmask, levels=[0.5], colors='gold', linewidths=1.0)
        ax.set_xticks([]); ax.set_yticks([])

    # memory m at t=600
    ax_m = fig.add_subplot(gs[row, 4])
    m_final = snap_at(sol_rec, 600, T_PRE, T_PRE+T_ABL)[NC+NPAIRS:].reshape(N, N)
    im_m = ax_m.imshow(m_final, cmap='viridis', vmin=0, vmax=1,
                        origin='upper', interpolation='nearest')
    ax_m.set_xticks([]); ax_m.set_yticks([])
    if row == 0:
        plt.colorbar(im_m, ax=ax_m, shrink=0.85, pad=0.04,
                     label='$m_i$').ax.tick_params(labelsize=7)

# shared V colourbar
cax = fig.add_axes([0.955, 0.08, 0.013, 0.80])
sm  = plt.cm.ScalarMappable(cmap=CMAP_V, norm=plt.Normalize(Vm, Vp))
cb  = fig.colorbar(sm, cax=cax)
cb.set_label(r'$V_i$', fontsize=9); cb.ax.tick_params(labelsize=7)

fig.suptitle(
    r'2D recovery from three structured-perturbation geometries  '
    r'($\alpha=1.2>\alpha_c^*$, $N=20\times20$)',
    fontsize=10, y=0.995, style='italic', color='#333333')

outpath = 'figs/fig_2d_lesions.pdf'
fig.savefig(outpath, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {outpath}')
print('Done.')
