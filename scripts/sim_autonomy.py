"""
sim_autonomy.py
===============
Coupling-controlled persistence and adaptive isolation of local attractor autonomy (fig_autonomy.pdf).

Panel A: recovery of an inverted block Omega versus boundary conductance G_bnd, memory and
         conductances frozen, internal conductance fixed at 0.5 (|Omega| > 1).
Panel B: spectrum of f'(V_alt) + alpha + lambda_k(L_Gint) for |Omega| = 7 (alternative branch is stable).
Panels C, D: full adaptive dynamics (G and m evolve) for the two orientations at matched initial
         conductance: internal and boundary conductance, and mean target phi(m_i) over Omega.

Usage: python3 sim_autonomy.py          (about 6 minutes)
       QUICK=1 python3 sim_autonomy.py  (smoke test)
"""
import os, sys, time
import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

QUICK = bool(os.environ.get("QUICK"))
T_A = 30.0 if QUICK else 300.0          # horizon must exceed the front-crossing time of the largest block
T_ADAPT = 40.0 if QUICK else 200.0
t0 = time.time()
SEED = 1      # pattern with a 21-cell hyperpolarized and a 7-cell depolarized domain; thresholds are identical over 16 seeds (sim_seeds.py)
p = mm.MinimalParams(N=60, seed=SEED)
net = mm.MinimalNetwork(p); net.run(300.0, record_every=300.0)
V_ref, m_ref, G_ref, n_ref = net.V.copy(), net.m.copy(), net.G.copy(), net.n.copy()
phi = lambda m: mm.phi_func(m, p.V_minus, p.V_plus, p.k_phi)
fprime = lambda V: p.a_bist * ((V - p.V_0) * (V - p.V_plus) + (V - p.V_minus) * (V - p.V_plus) + (V - p.V_minus) * (V - p.V_0))

G_INT = 0.5


def domains(V):
    sg = (V > 0).astype(int); out = []; start = 0
    for i in range(1, len(sg) + 1):
        if i == len(sg) or sg[i] != sg[start]: out.append((start, i - 1, int(sg[start]))); start = i
    return out


def block(dom, K):
    c = (dom[0] + dom[1]) // 2; h = (K - 1) // 2; om = list(range(c - h, c + h + 1)); lo, hi = om[0], om[-1]
    return dict(OMEGA=om, boundary_edges=[(lo - 1, lo), (hi, hi + 1)], internal_edges=[(i, i + 1) for i in range(lo, hi)])


DOMS = domains(V_ref)
HYPER = max([d for d in DOMS if d[2] == 0], key=lambda d: d[1] - d[0]); DEPOL = max([d for d in DOMS if d[2] == 1], key=lambda d: d[1] - d[0])
configs = {K: block(HYPER, K) for K in (1, 3, 7)}
Gbnd_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


def run_frozen(Vg, mg, Gg, n0, T):
    net2 = mm.MinimalNetwork(p); net2.V, net2.m, net2.G, net2.n = Vg, mg, Gg, n0
    net2.time = 0.0; net2.hist_t, net2.hist_V, net2.hist_m, net2.hist_G, net2.hist_n = [], [], [], [], []; net2._next_n_update = 1e9
    orig = net2._derivatives
    net2._derivatives = lambda V, m, G, n: (lambda r: (r[0], np.zeros_like(r[1]), np.zeros_like(r[2])))(orig(V, m, G, n))
    net2.run(T, record_every=T)
    return net2


panelA = {}
for K, cfg in configs.items():
    Es = []
    for Gb in Gbnd_grid:
        Vg, mg, Gg = V_ref.copy(), m_ref.copy(), G_ref.copy()
        for i in cfg['OMEGA']: Vg[i] = -Vg[i]
        for (i, j) in cfg['boundary_edges']: Gg[i, j] = Gg[j, i] = Gb
        for (i, j) in cfg['internal_edges']: Gg[i, j] = Gg[j, i] = G_INT
        Es.append(np.mean(np.abs(run_frozen(Vg, mg, Gg, n_ref.copy(), T_A).V[cfg['OMEGA']] - V_ref[cfg['OMEGA']])))
    panelA[K] = np.array(Es)
    idx = np.where(panelA[K] < 0.3)[0]
    print(f"[{time.time()-t0:.0f}s] |Omega|={K}: E = {np.round(Es, 3)}; recovery threshold in "
          f"({Gbnd_grid[idx[0]-1] if len(idx) and idx[0] > 0 else None}, {Gbnd_grid[idx[0]] if len(idx) else None}]", flush=True)

# Panel B: Hessian spectrum for |Omega| = 7 at G_int = 0.5
O7 = configs[7]['OMEGA']; n7 = len(O7); L7 = np.zeros((n7, n7))
for (i, j) in configs[7]['internal_edges']:
    a_, b_ = O7.index(i), O7.index(j)
    L7[a_, a_] += G_INT; L7[b_, b_] += G_INT; L7[a_, b_] -= G_INT; L7[b_, a_] -= G_INT
eig_J = fprime(1.477) + p.alpha + np.linalg.eigvalsh(L7)
print("Panel B eigenvalues:", np.round(eig_J, 3))

# Panels C, D: adaptive dynamics, matched initial conductance, both orientations
conditions = {
    'A': dict(label=r'$V_-\to V_+$', color='crimson', **block(HYPER, 3)),
    'B': dict(label=r'$V_+\to V_-$', color='steelblue', **block(DEPOL, 3)),
}


def run_adaptive(cfg, G0=0.5, T=200.0, dt_log=2.0):
    O, bnd, intl = cfg['OMEGA'], cfg['boundary_edges'], cfg['internal_edges']
    Vg, mg, Gg = V_ref.copy(), m_ref.copy(), G_ref.copy()
    for i in O: Vg[i] = -Vg[i]
    for (i, j) in bnd + intl: Gg[i, j] = Gg[j, i] = G0
    net2 = mm.MinimalNetwork(p); net2.V, net2.m, net2.G, net2.n = Vg, mg, Gg, n_ref.copy(); net2.time = 0.0; net2._next_n_update = 1e9
    log = dict(t=[], Gint=[], Gbnd=[], phi_mean=[]); t = 0.0; nxt = 0.0
    for _ in range(int(T / p.dt)):
        net2._rk4_step(); t += p.dt
        if t >= nxt:
            log['t'].append(t); log['Gint'].append(np.mean([net2.G[i, j] for (i, j) in intl])); log['Gbnd'].append(np.mean([net2.G[i, j] for (i, j) in bnd]))
            log['phi_mean'].append(np.mean(phi(net2.m[O]))); nxt += dt_log
    return {k: np.array(v) for k, v in log.items()}


logs = {k: run_adaptive(c, T=T_ADAPT) for k, c in conditions.items()}
for k, c in conditions.items():
    L = logs[k]; print(f"{c['label']} (Omega {c['OMEGA']}): final Gint={L['Gint'][-1]:.4f} Gbnd={L['Gbnd'][-1]:.4f} phi_mean {L['phi_mean'][0]:.3f} -> {L['phi_mean'][-1]:.3f}")

fig = plt.figure(figsize=(11, 8)); gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)
axA = fig.add_subplot(gs[0, 0])
for K, c in zip([1, 3, 7], ['seagreen', 'darkorange', 'purple']):
    axA.plot(Gbnd_grid, panelA[K], 'o-', color=c, lw=1.8, ms=4, label=fr'$|\Omega|={K}$')
axA.axhline(0.3, color='grey', ls=':', lw=0.8)
axA.set_xlabel(r'boundary conductance $G_{\partial\Omega}$ (per edge)'); axA.set_ylabel(r'$E_\Omega$ (deviation from $V^*|_\Omega$)')
axA.set_title('(A) Recovery vs. boundary conductance', fontsize=10.5); axA.legend(fontsize=8.5)
axB = fig.add_subplot(gs[0, 1]); axB.stem(np.arange(1, n7 + 1), eig_J, basefmt=' '); axB.axhline(0, color='k', lw=0.8)
axB.set_xlabel('mode index $k$'); axB.set_ylabel(r"$f'(V_{\rm alt})+\alpha+\lambda_k(L_{G_{\rm int}})$")
axB.set_title(r'(B) Hessian spectrum, alternative branch, $|\Omega|=7$', fontsize=10.5)
axC = fig.add_subplot(gs[1, 0])
for k, c in conditions.items():
    L = logs[k]
    axC.plot(L['t'], L['Gint'], '-', color=c['color'], lw=1.6, label=f"{c['label']}: $\\bar G_{{\\rm int}}$")
    axC.plot(L['t'], L['Gbnd'], '--', color=c['color'], lw=1.6, label=f"{c['label']}: $\\bar G_{{\\partial\\Omega}}$")
axC.set_xlabel('time'); axC.set_ylabel('conductance'); axC.set_title('(C) Adaptive isolation: internal vs. boundary', fontsize=10.5); axC.legend(fontsize=7.5)
axD = fig.add_subplot(gs[1, 1])
for k, c in conditions.items(): axD.plot(logs[k]['t'], logs[k]['phi_mean'], '-', color=c['color'], lw=1.8, label=c['label'])
axD.axhline(p.V_plus, color='grey', ls=':', lw=0.8); axD.axhline(p.V_minus, color='grey', ls=':', lw=0.8)
axD.set_xlabel('time'); axD.set_ylabel(r'$\overline{\varphi(m_i)}$ over $\Omega$'); axD.set_title('(D) Memory re-consolidation', fontsize=10.5); axD.legend(fontsize=8.5)
os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
out = os.path.join(HERE, 'figs', 'fig_autonomy_quick.pdf' if QUICK else 'fig_autonomy.pdf')
fig.savefig(out, dpi=180, bbox_inches='tight'); print("saved", out)
