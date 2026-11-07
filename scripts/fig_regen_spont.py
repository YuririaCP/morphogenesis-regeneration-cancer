"""
fig_regen_spont.py
==================
New Figure 6 (working version): regeneration from the spontaneously formed pattern of the morphogenesis and
cancer protocols (seed 7, T1 = 300). A 3-cell lesion (cells 16-18) interior to the 5-cell depolarized domain
(cells 15-19) is imposed and followed for 8 time units.

(a) pre-lesion pattern: V_i (bars) and m_i (dots)
(b) space-time plot of V_i after the lesion (white = vacant)
(c) occupied fraction of the lesion versus time for the nominal model and four ablations
"""
import os, sys
import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

SEED, LO, HI, T_END, DT = 7, 16, 19, 4.0, 0.1
mpl = matplotlib
mpl.rcParams.update({'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6})
VARIANTS = {"nominal": {}, r"$\kappa=0$": {"kappa": 0.0}, r"$\alpha=0$": {"alpha": 0.0},
            r"$\alpha=\kappa=0$": {"alpha": 0.0, "kappa": 0.0}, r"$\alpha=\kappa=\lambda_m=0$": {"alpha": 0.0, "kappa": 0.0, "lam_m": 0.0}}

p0 = mm.MinimalParams(N=60, seed=SEED); base = mm.MinimalNetwork(p0); base.run(300.0, record_every=300.0)
V0, m0, G0, n0 = base.V.copy(), base.m.copy(), base.G.copy(), base.n.copy()


def run_variant(over):
    p = mm.MinimalParams(N=60, seed=SEED)
    for k, v in over.items(): setattr(p, k, v)
    net = mm.MinimalNetwork(p); net.V, net.m, net.G, net.n = V0.copy(), m0.copy(), G0.copy(), n0.copy()
    net.time = 0.0; net._next_n_update = 0.0
    net.impose_lesion(LO, HI)
    ts, Vs, ns, occ = [0.0], [net.V.copy()], [net.n.copy()], [float(net.n[LO:HI].mean())]
    for k in range(int(round(T_END / DT))):
        net.run(DT, record_every=DT)
        ts.append((k + 1) * DT); Vs.append(net.V.copy()); ns.append(net.n.copy()); occ.append(float(net.n[LO:HI].mean()))
    return np.array(ts), np.array(Vs), np.array(ns), np.array(occ)

res = {name: run_variant(over) for name, over in VARIANTS.items()}
ts, Vs, ns, _ = res["nominal"]

fig = plt.figure(figsize=(7.0, 4.3), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.0], width_ratios=[1.15, 1.0])

# (a) pre-lesion pattern
ax = fig.add_subplot(gs[0, :])
cols = np.where(V0 > 0, '#b2182b', '#2166ac')
ax.bar(np.arange(60), V0, color=cols, width=0.9, edgecolor='none')
ax.axvspan(LO - 0.5, HI - 0.5, color='gold', alpha=0.35, lw=0)
ax.set_xlim(-0.8, 59.8); ax.set_ylim(-2.0, 2.0)
ax.set_xlabel('cell index $i$'); ax.set_ylabel('$V_i$')
ax2 = ax.twinx(); ax2.plot(np.arange(60), m0, 'k.', ms=2.2); ax2.set_ylim(-0.1, 1.1); ax2.set_ylabel('$m_i$')
ax2.tick_params(labelsize=6)
dm = []
s = 0
for i in range(1, 61):
    if i == 60 or (V0[i] > 0) != (V0[s] > 0): dm.append((s, i - 1, V0[s] > 0)); s = i
for a_, b_, pos in dm: ax.text((a_ + b_) / 2, 1.62, '+' if pos else '$-$', ha='center', va='center', fontsize=7, fontweight='bold')
ax.set_title('(a) Spontaneous pattern before the lesion (seed 7, $t=300$); lesion $\\Omega$ shaded', loc='left')

# (b) space-time
axb = fig.add_subplot(gs[1, 0])
c0, c1 = 8, 28
Vm = np.where(ns[:, c0:c1] == 1, Vs[:, c0:c1], np.nan)
im = axb.pcolormesh(np.arange(c0, c1 + 1) - 0.5, np.append(ts, ts[-1] + DT) - DT / 2, np.ma.masked_invalid(Vm), cmap='RdBu_r', vmin=-1.5, vmax=1.5, shading='flat', edgecolors='none', rasterized=True)
axb.set_facecolor('white')
axb.axvspan(LO - 0.5, HI - 0.5, ymin=0, ymax=1, facecolor='none', edgecolor='gold', lw=0.8)
axb.set_xlabel('cell index $i$'); axb.set_ylabel('time after lesion')
cb = fig.colorbar(im, ax=axb, shrink=0.85, pad=0.01, aspect=22); cb.set_label('$V_i$'); cb.ax.tick_params(labelsize=5.5)
axb.set_title('(b) $V_i(t)$ after the lesion (white: vacant)', loc='left')

# (c) occupancy of lesion vs time
axc = fig.add_subplot(gs[1, 1])
mk = ['o', 's', '^', 'v', 'D']; off = np.linspace(-0.03, 0.03, 5)
for (name, (t_, V_, n_, occ)), m_, o_ in zip(res.items(), mk, off):
    axc.plot(t_ + o_, occ, marker=m_, ms=2.2, lw=0.7, ls='-', label=name, markevery=5)
axc.set_xlabel('time after lesion'); axc.set_ylabel('occupied fraction of $\\Omega$')
axc.set_ylim(-0.05, 1.08); axc.set_xlim(0, T_END)
axc.legend(loc='lower right', frameon=False, ncol=1)
axc.text(2.1, 0.45, 'the five curves coincide', fontsize=6)
axc.set_title('(c) Occupancy of $\\Omega$: nominal and ablated models', loc='left')
for a_ in (ax, axb, axc): a_.tick_params(length=2)

out = os.path.join(HERE, '..', 'figs', 'fig_regeneration.pdf')
fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out)
t_full = ts[np.argmax(res['nominal'][3] >= 1.0)]
print('nominal recovered at t =', t_full, '; variants occ at t=2:', {k: float(v[3][int(2/DT)]) for k, v in res.items()})
