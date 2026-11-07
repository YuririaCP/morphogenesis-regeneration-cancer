"""
sim_blockade.py
===============
Transient coupling blockade of a small region and its outcome as a function of the blockade duration D
(support for Prediction 3 of the main text).

Protocol (1D chain, N = 60, pattern formed for T = 300 under the full dynamics):
  t = 0   : the voltages of a block Omega of 3 cells, centred in the largest domain of a given polarity
            (>= 5 cells), are inverted, and the TRANSMISSION through every edge incident to Omega (internal and
            boundary) is blocked: the conductance entering the voltage and memory equations is multiplied by
            EPS_BLOCK = 0.03, while the stored conductance G keeps following its adaptive rule (a drug that
            closes the channels does not erase the junction state).
  t = D   : the block is released; everything acts freely.
  t = D+T_AFTER : outcome.
Outcomes: the sign of Omega relative to the reference pattern (recovered / persists / mixed), the mean target
phi(m) over Omega at release (memory shift) and at the end.  D = 0 is the control without blockade
(inversion only, conductances as in the pinned pattern).

Usage: python3 sim_blockade.py            (full sweep, 16 seeds)
       QUICK=1 python3 sim_blockade.py    (pilot: 2 seeds, short horizon)
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

QUICK = bool(os.environ.get("QUICK"))
SEEDS = [1, 2] if QUICK else list(range(1, 17))
DS = [0.0, 10.0, 20.0, 40.0, 60.0, 100.0] if QUICK else [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0]
T_AFTER = 200.0 if QUICK else 300.0
EPS_BLOCK = 0.03
OUT = os.path.join(HERE, "blockade_results_quick.jsonl" if QUICK else "blockade_results.jsonl")


def reference(seed):
    p = mm.MinimalParams(N=60, seed=seed); net = mm.MinimalNetwork(p); net.run(300.0, record_every=300.0)
    return net.V.copy(), net.m.copy(), net.G.copy(), net.n.copy()


def domains(V):
    sg = (V > 0).astype(int); out = []; start = 0
    for i in range(1, len(sg) + 1):
        if i == len(sg) or sg[i] != sg[start]:
            out.append((start, i - 1, int(sg[start]))); start = i
    return out


def centred_block(dom, K):
    a, b, _ = dom; c = (a + b) // 2; h = (K - 1) // 2
    return list(range(c - h, c + h + 1))


def one_run(args):
    seed, name, pol, D = args
    V_ref, m_ref, G_ref, n_ref = reference(seed)
    cand = [d for d in domains(V_ref) if d[2] == pol and d[1] - d[0] + 1 >= 5]
    if not cand:
        return dict(seed=seed, orient=name, D=D, skipped=True)
    d = max(cand, key=lambda x: x[1] - x[0]); om = centred_block(d, 3); lo, hi = om[0], om[-1]
    edges = [(i, j) for (i, j) in zip(range(59), range(1, 60)) if (i in om or j in om)]
    scale = np.ones((60, 60))
    for (i, j) in edges: scale[i, j] = scale[j, i] = EPS_BLOCK
    net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=seed))
    V0 = V_ref.copy()
    for i in om: V0[i] = -V0[i]
    net.V, net.m, net.G, net.n = V0, m_ref.copy(), G_ref.copy(), n_ref.copy()
    net.time = 0.0; net._next_n_update = 1e9
    phi = lambda m: mm.phi_func(m, net.p.V_minus, net.p.V_plus, net.p.k_phi)
    dt = net.p.dt
    orig = net._derivatives
    def blocked(V, m, G, n):
        dVe, dme, _ = orig(V, m, G * scale, n)
        _, _, dGt = orig(V, m, G, n)
        return dVe, dme, dGt
    if D > 0:
        net._derivatives = blocked
        for _ in range(int(round(D / dt))): net._rk4_step()
        net._derivatives = orig
    phi_rel = float(np.mean(phi(net.m[om])))
    V_rel = float(np.mean(net.V[om]))
    for _ in range(int(round(T_AFTER / dt))):
        net._rk4_step()
    s_ref = np.sign(V_ref[om]); s_end = np.sign(net.V[om])
    n_same = int(np.sum(s_ref == s_end))
    outcome = "recovered" if n_same == 3 else ("persists" if n_same == 0 else "mixed")
    return dict(seed=seed, orient=name, D=D, domain=[d[0], d[1]], omega=[lo, hi], outcome=outcome,
                phi_ref=float(np.mean(phi(m_ref[om]))), phi_release=phi_rel, V_release=V_rel,
                phi_end=float(np.mean(phi(net.m[om]))), V_end=float(np.mean(net.V[om])),
                G_bnd_end=float(np.mean([net.G[lo - 1, lo], net.G[hi, hi + 1]])))


if __name__ == "__main__":
    jobs = [(s, name, pol, D) for s in SEEDS for (name, pol) in (("V+ to V-", 1), ("V- to V+", 0)) for D in DS]
    t0 = time.time()
    with Pool() as pool, open(OUT, "w") as f:
        for k, r in enumerate(pool.imap_unordered(one_run, jobs)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if (k + 1) % 20 == 0: print(f"[{time.time()-t0:6.0f}s] {k+1}/{len(jobs)}", flush=True)
    print("results written to", OUT)
