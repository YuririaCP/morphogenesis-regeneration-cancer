"""
sim_errorcorr_1d.py
===================
One-dimensional error correction after structured pattern inversion (Supplementary Fig. S3).

Protocol (1D chain, N = 60, seed 7, full adaptive dynamics): the spontaneous pattern is formed for
T_PRE = 150 time units with the nominal parameters. At t = T_PRE the voltages of the connected region
Omega = cells 22..37 (inside the large hyperpolarized domain) are inverted, V_i <- -V_i, with memory and
conductances unchanged, and the error-correction strength is set to alpha_3 in {0.80, 0.30}; the system
then evolves for T_POST further time units. The error is
    E_Omega(t) = |Omega|^-1 sum_{i in Omega} |V_i(t) - V_i^*|,
with V_i^* the voltage just before the inversion.

Usage: python3 sim_errorcorr_1d.py
"""
import os, sys
import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

T_PRE, T_POST, REC = 150.0, 60.0, 1.0
OMEGA = list(range(22, 38))
ALPHAS = {'a': 0.80, 'b': 0.30}
ALPHA_C = 0.5625


def run(alpha3):
    p = mm.MinimalParams(N=60, seed=7)
    net = mm.MinimalNetwork(p)
    net.run(T_PRE, record_every=T_PRE)
    V_star = net.V.copy()
    net.V[OMEGA] = -net.V[OMEGA]
    net.p.alpha = alpha3
    t_rec, V_rec, E = [0.0], [net.V.copy()], [float(np.mean(np.abs(net.V[OMEGA] - V_star[OMEGA])))]
    steps = int(round(T_POST / p.dt)); every = int(round(REC / p.dt))
    for k in range(1, steps + 1):
        if net.time >= net._next_n_update: net._update_occupancy(); net._next_n_update += p.dt_n
        net._rk4_step(); net.time += p.dt
        if k % every == 0:
            t_rec.append(k * p.dt); V_rec.append(net.V.copy()); E.append(float(np.mean(np.abs(net.V[OMEGA] - V_star[OMEGA]))))
    return np.array(t_rec), np.array(V_rec), np.array(E), V_star


def pre_run():
    p = mm.MinimalParams(N=60, seed=7); net = mm.MinimalNetwork(p)
    net.run(T_PRE, record_every=1.0)
    return np.array(net.hist_t), np.array(net.hist_V)


if __name__ == "__main__":
    res = {k: run(a) for k, a in ALPHAS.items()}
    tp, Vp = pre_run()
    for k, (t, V, E, Vs) in res.items():
        print(f"alpha_3={ALPHAS[k]:.2f}: E(0)={E[0]:.2f}, E(T_POST)={E[-1]:.3f}, final mean V over Omega={V[-1][OMEGA].mean():+.2f}")
    fig = plt.figure(figsize=(8.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    axes_top = []
    for col, k in enumerate(('a', 'b')):
        ax = fig.add_subplot(gs[0, col]); t, V, E, Vs = res[k]
        T = np.concatenate([tp, T_PRE + t[1:]]); H = np.vstack([Vp, V[1:]])
        im = ax.imshow(H.T, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5,
                       extent=[T[0], T[-1], -0.5, 59.5], interpolation='nearest')
        ax.axvline(T_PRE, color='gold', ls='--', lw=1.2, label='pattern inversion')
        ax.axhline(OMEGA[0] - 0.5, color='gold', ls=':', lw=0.8); ax.axhline(OMEGA[-1] + 0.5, color='gold', ls=':', lw=0.8)
        rel = '>' if ALPHAS[k] > ALPHA_C else '<'
        verdict = 'corrects' if k == 'a' else 'alternative state persists'
        ax.set_title(rf'({k}) $\alpha_3={ALPHAS[k]:.2f}{rel}\alpha_c^*$: {verdict}', fontsize=8)
        ax.set_xlabel('time'); ax.set_ylabel('cell index $i$')
        ax.legend(loc='upper left', fontsize=6.5, framealpha=0.9)
        axes_top.append(ax)
    cb = fig.colorbar(im, ax=axes_top, pad=0.02, shrink=0.9); cb.set_label('$V_i$', fontsize=8)
    axc = fig.add_subplot(gs[1, :])
    axc.plot(res['a'][0], res['a'][2], color='#1f5fa8', lw=1.8, label=r'$\alpha=0.80$: correction succeeds')
    axc.plot(res['b'][0], res['b'][2], color='#c0392b', lw=1.8, ls='--', label=r'$\alpha=0.30$: alternative state persists')
    axc.set_xlabel('time since perturbation'); axc.set_ylabel(r'$E_\Omega(t)$')
    axc.set_title(r'Error relative to the pinned morphostatic pattern $V_i^*$', fontsize=9)
    axc.legend(loc='center right', fontsize=7)
    out = os.path.join(HERE, '..', 'figs', 'fig_error_correction.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out)
