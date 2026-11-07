"""
sim_noise.py
============
Noise experiments of the stochastic subsection (figure fig_noise_robustness.pdf).

Fast voltage layer, committed binary memory, uniform coupling G, 40-cell chain:
  dV_i = [-f(V_i) - alpha (V_i - phi(m_i)) + G sum_j (V_j - V_i)] dt + sigma dW_i

Experiments
  1. initial-condition control: same random initial conditions, sigma = 0 versus 0.05
  2. noise-induced escape of one inverted cell from the alternative branch (first passage with
     hysteresis), Arrhenius fit, comparison with the single-cell reduced barrier
  3. width of the weak-noise window below the saddle-node coupling
  4. energy step versus energy fluctuation at an escape

Usage: python3 sim_noise.py          (about 5 minutes on 4 cores)
       QUICK=1 python3 sim_noise.py  (smoke test)
"""
import os, time
from functools import lru_cache
import numpy as np
from multiprocessing import Pool
from scipy.optimize import brentq
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

QUICK = bool(os.environ.get("QUICK"))
HERE = os.path.dirname(os.path.abspath(__file__))
N, dt = 40, 0.02
a, Vm, V0, Vp, kphi = 1.0, -1.5, 0.0, 1.5, 6.0
phi = lambda m: Vm + (Vp - Vm) / (1 + np.exp(-kphi * (m - 0.5)))
m_commit = np.zeros(N); m_commit[:N // 2] = 1.0
phim = phi(m_commit)
f = lambda V: a * (V - Vm) * (V - V0) * (V - Vp)
CELL, THETA, ALPHA = 30, 0.5, 0.30
s1 = Vm + V0 + Vp; s2 = Vm * V0 + Vm * Vp + V0 * Vp; s3 = Vm * V0 * Vp
F = lambda V: a * (V**4 / 4 - s1 / 3 * V**3 + s2 / 2 * V**2 - s3 * V)
fp = lambda V: a * ((V - V0) * (V - Vp) + (V - Vm) * (V - Vp) + (V - Vm) * (V - V0))


def lap(V, G):
    out = np.zeros_like(V)
    out[..., :-1] += G * (V[..., 1:] - V[..., :-1]); out[..., 1:] += G * (V[..., :-1] - V[..., 1:])
    return out


def energy(V, G, alpha):
    return (F(V).sum(-1) + 0.5 * alpha * ((V - phim)**2).sum(-1) + 0.5 * G * ((V[..., 1:] - V[..., :-1])**2).sum(-1)) / N


@lru_cache(maxsize=None)
def equilibrium(G, alpha):
    V = np.where(phim > 0, Vp, Vm)[None, :].astype(float)
    for _ in range(20000):
        V = V + (-f(V) - alpha * (V - phim) + lap(V, G)) * dt
    V = V[0]; V.setflags(write=False)
    return V


def em_step(V, G, alpha, sigma, rng):
    V = V + (-f(V) - alpha * (V - phim) + lap(V, G)) * dt + sigma * np.sqrt(dt) * rng.standard_normal(V.shape)
    return np.clip(V, Vm - 0.5, Vp + 0.5)


def escape(G, alpha, sigma, M, T, seed):
    Vstar = equilibrium(G, alpha); rng = np.random.default_rng(seed)
    V = np.tile(Vstar, (M, 1)); V[:, CELL] = -Vstar[CELL]
    tfp = np.full(M, np.inf)
    for s in range(int(T / dt)):
        V = em_step(V, G, alpha, sigma, rng)
        hit = (V[:, CELL] < -THETA) & np.isinf(tfp); tfp[hit] = (s + 1) * dt
        if not np.isinf(tfp).any(): break
    return tfp


def reduced(G, alpha):
    """Barrier of the single-cell potential for the inverted cell, neighbours at their equilibrium voltages."""
    Vs = equilibrium(G, alpha); phi_c = phim[CELL]; Vl, Vr = Vs[CELL - 1], Vs[CELL + 1]
    U = lambda V: F(V) + 0.5 * alpha * (V - phi_c)**2 + 0.5 * G * ((V - Vl)**2 + (V - Vr)**2)
    dU = lambda V: f(V) + alpha * (V - phi_c) + G * ((V - Vl) + (V - Vr))
    grid = np.linspace(-2.2, 2.2, 8801); d = dU(grid)
    roots = [brentq(dU, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if d[i] * d[i + 1] < 0]
    if len(roots) < 3: return None
    Valt, Vsad = roots[-1], roots[1]
    return U(Vsad) - U(Valt), fp(Valt) + alpha + 2 * G, abs(fp(Vsad) + alpha + 2 * G)


def kramers(G, sigma):
    dU, k1, k2 = reduced(G, ALPHA)
    return 2 * np.pi / np.sqrt(k1 * k2) * np.exp(2 * dU / sigma**2)


def _job(args):
    G, sigma, M, TMAX = args
    tfp = escape(G, ALPHA, sigma, M, TMAX, seed=int(1000 * G + 100 * sigma))
    ev = np.isfinite(tfp)
    return G, sigma, int(ev.sum()), float(np.sum(np.where(ev, tfp, TMAX)))


if __name__ == "__main__":
    t0 = time.time()
    M_IC = 20 if QUICK else 200
    # ---- 1. initial-condition control --------------------------------------------------------
    print("== 1. same initial conditions, sigma=0 vs 0.05 ==")
    for G, al, name in [(0.55, 0.30, 'above'), (0.15, 0.35, 'near'), (0.02, 0.05, 'below')]:
        ic = np.random.default_rng(7).uniform(Vm, Vp, (M_IC, N)); out = {}
        for sigma in (0.0, 0.05):
            rng = np.random.default_rng(11); V = ic.copy()
            for _ in range(int(80.0 / dt)):
                V = em_step(V, G, al, sigma, rng)
            out[sigma] = V
        nd = (np.sign(out[0.0]) != np.sign(out[0.05])).sum(1); mis = (np.sign(out[0.0]) != np.sign(phim)).sum(1)
        print(f"{name:6s} identical pattern {np.mean(nd==0)*100:5.1f}%  mean cells differing {nd.mean():.2f}  "
              f"misaligned cells {mis.min()}-{mis.max()}")
    # ---- 2. escape ensemble ------------------------------------------------------------------
    GS = [0.05, 0.10] if QUICK else [0.03, 0.05, 0.07, 0.10]
    SIG = [0.4, 0.5] if QUICK else [0.25, 0.30, 0.35, 0.40, 0.50]
    M, TMAX = (20, 300.0) if QUICK else (200, 3000.0)
    with Pool(4) as p:
        res = p.map(_job, [(G, s, M, TMAX) for G in GS for s in SIG])
    G_, S_, NE, EX = (np.array(x) for x in zip(*res))
    print("== 2. mean escape times (G, sigma, events, tau) ==")
    for g, s, n, e in res: print(f"{g:4.2f} {s:4.2f} {n:4d}/{M}  {e/max(n,1):10.1f}")
    fit = []
    for g in GS:
        mk = (G_ == g) & (NE >= (5 if QUICK else 20))
        if mk.sum() >= 2:
            sl = np.polyfit(1 / S_[mk]**2, np.log(EX[mk] / NE[mk]), 1)[0]; fit.append((g, sl / 2))
            print(f"G={g:.2f}: effective barrier {sl/2:.3f}  reduced barrier {reduced(g, ALPHA)[0]:.3f}")
    # ---- 3. saddle-node coupling and weak-noise window ---------------------------------------
    lo, hi = 0.10, 0.20
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if reduced(mid, ALPHA) is None: hi = mid
        else: lo = mid
    gc = lo; print(f"== 3. reduced saddle-node coupling G_c = {gc:.4f}")
    for sigma, Tobs in ((0.05, 80.0), (0.05, 1000.0), (0.10, 80.0), (0.20, 80.0)):
        for g in np.linspace(0.05, gc - 1e-5, 300):
            if reduced(g, ALPHA) is not None and kramers(g, sigma) <= Tobs:
                print(f"sigma={sigma}, T={Tobs:g}: escape requires G >= {g:.4f} ({(gc-g)/gc*100:.1f}% below G_c)"); break
    # ---- 4. trajectories and energy ----------------------------------------------------------
    G, sigma, Mt, T = 0.05, 0.40, (40 if QUICK else 400), 200.0
    Vs = equilibrium(G, ALPHA); rng = np.random.default_rng(5)
    V = np.tile(Vs, (Mt, 1)); V[:, CELL] = -Vs[CELL]; ts, V30, H = [], [], []
    for s in range(int(T / dt)):
        V = em_step(V, G, ALPHA, sigma, rng)
        if s % int(0.5 / dt) == 0: ts.append(s * dt); V30.append(V[:, CELL].copy()); H.append(energy(V, G, ALPHA))
    ts, V30, H = np.array(ts), np.array(V30), np.array(H); Hstar = energy(Vs[None], G, ALPHA)[0]
    fp_t = np.array([ts[np.where(V30[:, j] < -THETA)[0][0]] if np.any(V30[:, j] < -THETA) else np.inf for j in range(Mt)])
    pre = np.concatenate([H[5:np.searchsorted(ts, fp_t[j]) - 4, j] for j in range(Mt) if 40 < fp_t[j] < np.inf])
    post = np.array([H[np.searchsorted(ts, fp_t[j]) + 20:, j].mean() for j in range(Mt) if fp_t[j] < T - 30])
    print(f"== 4. H/N waiting phase std {pre.std():.4f}, peak-to-peak {np.ptp(pre):.4f}; step at escape {pre.mean()-post.mean():.4f}")
    # ---- figure ------------------------------------------------------------------------------
    j_esc = int(np.where((fp_t > 60) & (fp_t < 100))[0][0]) if np.any((fp_t > 60) & (fp_t < 100)) else int(np.argmin(np.abs(fp_t - 80)))
    j_stay = int(np.where(np.isinf(fp_t))[0][0]) if np.any(np.isinf(fp_t)) else int(np.argmax(fp_t))
    fig = plt.figure(figsize=(10.4, 4.3)); gs = gridspec.GridSpec(2, 3, hspace=0.12, wspace=0.36)
    axV = fig.add_subplot(gs[0, 0]); axH = fig.add_subplot(gs[1, 0], sharex=axV)
    for j, c, lab in ((j_esc, 'crimson', 'escapes'), (j_stay, 'steelblue', 'stays')):
        axV.plot(ts, V30[:, j], color=c, lw=0.9, label=lab); axH.plot(ts, H[:, j], color=c, lw=0.9)
    axV.axhline(THETA, color='k', ls=':', lw=0.8); axV.axhline(-THETA, color='k', ls=':', lw=0.8)
    axV.set_ylabel(r'$V_{30}$'); axV.set_title('(A) Perturbed cell and per-cell energy', fontsize=9.5)
    axV.legend(fontsize=7, loc='upper right'); plt.setp(axV.get_xticklabels(), visible=False)
    axH.axhline(Hstar, color='k', ls='--', lw=0.8); axH.set_ylabel(r'$\mathcal{H}/N$'); axH.set_xlabel('time')
    cols = {0.03: '#1b9e77', 0.05: '#d95f02', 0.07: '#7570b3', 0.10: '#e7298a'}
    axB = fig.add_subplot(gs[:, 1]); xs = np.linspace(2.5, 17, 50)
    for g in GS:
        mk = (G_ == g) & (NE >= (5 if QUICK else 20))
        if mk.sum() == 0: continue
        axB.semilogy(1 / S_[mk]**2, EX[mk] / NE[mk], 'o', color=cols[g], ms=4.5, label=fr'$\bar G={g:.2f}$')
        dU, k1, k2 = reduced(g, ALPHA)
        axB.semilogy(xs, 2 * np.pi / np.sqrt(k1 * k2) * np.exp(2 * dU * xs), '-', color=cols[g], lw=1)
    axB.set_xlabel(r'$1/\sigma^2$'); axB.set_ylabel('mean escape time'); axB.set_title('(B) Arrhenius dependence', fontsize=9.5)
    axB.legend(fontsize=7, loc='upper left')
    axC = fig.add_subplot(gs[:, 2]); gg = np.linspace(0.02, gc - 0.002, 80)
    axC.plot(gg, [reduced(g, ALPHA)[0] for g in gg], 'k-', lw=1.3, label='reduced single-cell barrier')
    axC.plot([x[0] for x in fit], [x[1] for x in fit], 'o', color='crimson', ms=5, label='fitted from escape times')
    axC.axvline(gc, color='grey', ls='--', lw=0.9); axC.axhline(0.004, color='steelblue', ls=':', lw=1)
    axC.text(0.02, 0.0065, r'barrier reachable at $\sigma=0.05$', fontsize=7, color='steelblue')
    axC.set_xlabel(r'uniform coupling $\bar G$'); axC.set_ylabel(r'effective barrier $\Delta U$')
    axC.set_title('(C) Barrier vanishes at the saddle-node', fontsize=9.5); axC.legend(fontsize=7, loc='upper right')
    os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
    out = os.path.join(HERE, 'figs', 'fig_noise_robustness_quick.pdf' if QUICK else 'fig_noise_robustness.pdf')
    fig.savefig(out, dpi=180, bbox_inches='tight'); print(f"saved {out}  [{time.time()-t0:.0f}s]")
