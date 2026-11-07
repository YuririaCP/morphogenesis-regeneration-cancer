"""
sim_regimes.py
==============
Recovery / autonomy diagrams in (G, alpha) space (figure fig_regimes.pdf).

From the pinned reference pattern (seed 7, T1 = 300) memory and conductances are frozen, every edge
is set to a uniform Gbar and the error-correction strength is alpha. The pattern is relaxed for
T = 100 to its reference state V*(Gbar, alpha); a connected block Omega of 1, 3, 7 or 15 cells inside
the hyperpolarized domain is inverted and the fast subsystem is integrated for T = 300. Output is the
fraction S of Omega whose sign differs from V* (0 recovered, 1 persists).

Usage: python3 sim_regimes.py          (about 15 minutes on 4 cores)
       QUICK=1 python3 sim_regimes.py  (smoke test)
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import brentq
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

QUICK = bool(os.environ.get("QUICK"))
ALPHAS = np.array([0.05, 0.30, 0.80]) if QUICK else np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80])
GBARS = np.array([0.0, 0.2, 0.5, 1.0]) if QUICK else np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
T_REL, T_REC = (20.0, 30.0) if QUICK else (100.0, 300.0)
OMEGA = {1: [30], 3: [29, 30, 31], 7: list(range(27, 34)), 15: list(range(23, 38))}


def reference():
    p = mm.MinimalParams(N=60, seed=7); net = mm.MinimalNetwork(p); net.run(300.0, record_every=300.0)
    return net.V.copy(), net.m.copy(), net.G.copy(), net.n.copy(), net.neighbors


def frozen_run(ref, alpha, gbar, V0, T):
    V_ref, m_ref, G_ref, n_ref, nbrs = ref
    net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=7, alpha=alpha))
    G = np.zeros_like(G_ref)
    for i in range(60):
        for j in nbrs[i]: G[i, j] = gbar
    net.V, net.m, net.G, net.n = V0.copy(), m_ref.copy(), G, n_ref.copy()
    net.time = 0.0; net._next_n_update = 1e9
    orig = net._derivatives
    net._derivatives = lambda V, m, G, n: (lambda r: (r[0], np.zeros_like(r[1]), np.zeros_like(r[2])))(orig(V, m, G, n))
    net.hist_t, net.hist_V, net.hist_m, net.hist_G, net.hist_n = [], [], [], [], []
    net.run(T, record_every=T)
    return net.V.copy()


def grid(K):
    ref = reference(); V_ref = ref[0]; om = OMEGA[K]
    S = np.full((len(ALPHAS), len(GBARS)), np.nan); ok = np.zeros(S.shape, bool)
    for ia, a in enumerate(ALPHAS):
        for ig, g in enumerate(GBARS):
            Vs = frozen_run(ref, a, g, V_ref, T_REL); ok[ia, ig] = np.all(np.sign(Vs) == np.sign(V_ref))
            V0 = Vs.copy(); V0[om] = -Vs[om]
            S[ia, ig] = np.mean(np.sign(frozen_run(ref, a, g, V0, T_REC)[om]) != np.sign(Vs[om]))
    return K, S, ok


L_, PHI, VN, AC = 1.5, -1.135, -1.477, 0.5625      # realised committed target and neighbour voltage


def refined(al, d=2):
    """Saddle-node coupling of one cell with d aligned neighbours at VN and committed target PHI."""
    disc = lambda G: -4 * (al + d * G - L_**2)**3 - 27 * (-al * PHI - d * G * VN)**2
    gs = np.linspace(0, 2, 4001); v = [disc(g) > 0 for g in gs]
    if not v[0]: return 0.0
    for i in range(1, len(gs)):
        if v[i - 1] and not v[i]: return brentq(disc, gs[i - 1], gs[i])
    return np.nan


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as pool: res = {K: (S, ok) for K, S, ok in pool.map(grid, [1, 3, 7, 15])}
    print(f"grid done [{time.time()-t0:.0f}s]; pattern preserved everywhere: {all(r[1].all() for r in res.values())}; "
          f"mixed outcomes: {sum(int(np.sum((r[0] > 0) & (r[0] < 1))) for r in res.values())}")

    def bracket(S, i):
        idx = np.where(S[i] < 0.5)[0]
        if len(idx) == 0: return np.nan, np.nan, np.nan
        lo = GBARS[idx[0] - 1] if idx[0] > 0 else 0.0; hi = GBARS[idx[0]]
        return lo, hi, 0.5 * (lo + hi)

    a_fine = np.linspace(0.04, 0.82, 200); cmap = ListedColormap(['#9ecae1', '#de2d26'])
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.9), gridspec_kw=dict(wspace=0.36))

    def panel(axx, S, title):
        ge = np.concatenate([[GBARS[0] - 0.01], 0.5 * (GBARS[1:] + GBARS[:-1]), [GBARS[-1] + 0.05]])
        ae = np.concatenate([[ALPHAS[0] - 0.025], 0.5 * (ALPHAS[1:] + ALPHAS[:-1]), [ALPHAS[-1] + 0.05]])
        axx.pcolormesh(ge, ae, S, cmap=cmap, vmin=0, vmax=1, shading='flat', edgecolor='white', linewidth=0.4)
        axx.set_xlabel(r'uniform coupling $\bar G$'); axx.set_ylabel(r'error correction $\alpha$'); axx.set_title(title, fontsize=10)
        axx.set_xlim(-0.01, 1.05); axx.set_ylim(0.025, 0.85)
        axx.plot([AC - a for a in a_fine], a_fine, ':', color='k', lw=1.1); axx.plot([(AC - a) / 2 for a in a_fine], a_fine, '--', color='k', lw=1.1)

    panel(ax[0], res[1][0], r'(A) $|\Omega|=1$'); ax[0].plot([refined(a) for a in a_fine], a_fine, 'k-', lw=1.4)
    panel(ax[1], res[7][0], r'(B) $|\Omega|=7$')
    ax[2].set_title('(C) Threshold vs. region size', fontsize=10)
    for K, c, mk in ((1, 'seagreen', 'o'), (3, 'darkorange', 's'), (7, 'purple', '^'), (15, 'k', 'x')):
        pts = [(bracket(res[K][0], i), a) for i, a in enumerate(ALPHAS)]; pts = [(b, a) for b, a in pts if not np.isnan(b[2]) and b[1] > 0]
        if pts:
            ax[2].errorbar([b[2] for b, a in pts], [a for b, a in pts], xerr=[[b[2] - b[0] for b, a in pts], [b[1] - b[2] for b, a in pts]],
                           fmt=mk, color=c, ms=4.5, lw=1, capsize=1.5, label=fr'$|\Omega|={K}$')
    ax[2].plot([refined(a) for a in a_fine], a_fine, '-', color='grey', lw=1.2)
    ax[2].set_xlabel(r'uniform coupling $\bar G$'); ax[2].set_ylabel(r'error correction $\alpha$')
    ax[2].set_xlim(-0.01, 1.05); ax[2].set_ylim(0.025, 0.85); ax[2].legend(fontsize=6.5, loc='upper right')
    ax[0].legend(handles=[Patch(color='#9ecae1', label='recovered'), Patch(color='#de2d26', label='persists'),
                          Line2D([], [], color='k', lw=1.4, label='single-cell prediction'),
                          Line2D([], [], color='k', ls='--', lw=1.1, label=r'$\alpha+2\bar G=\alpha_c^*$'),
                          Line2D([], [], color='k', ls=':', lw=1.1, label=r'$\alpha+\bar G=\alpha_c^*$')], fontsize=6, loc='upper right')
    os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
    out = os.path.join(HERE, 'figs', 'fig_regimes_quick.pdf' if QUICK else 'fig_regimes.pdf')
    fig.savefig(out, dpi=180, bbox_inches='tight'); print("saved", out)
