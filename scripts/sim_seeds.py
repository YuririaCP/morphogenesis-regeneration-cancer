"""
sim_seeds.py
============
Robustness of the 1D experiments of the Results section across random seeds (patterns differ in domain
number, size and polarity). Each experiment is placed according to the structure of the pattern of its seed.

  B1a recovery of an inverted block (1, 3, 7 cells) at the centre of the largest domain versus boundary
      conductance (internal conductance 0.5), frozen dynamics, T = 300
  B1b adaptive isolation: inversion of 3 cells in the largest hyperpolarized and the largest depolarized domain
      (>= 5 cells) at matched initial conductance 0.5, full adaptive dynamics, T = 200
  B3  recovery map over (alpha, Gbar) for blocks of 1, 3, 7 cells at the centre of the largest domain,
      uniform Gbar, frozen dynamics, relaxation T = 100, recovery T = 200

Results are appended to seeds_results.jsonl and summarised at the end.

Usage: python3 sim_seeds.py          (about 35 minutes on 4 cores)
       QUICK=1 python3 sim_seeds.py  (smoke test)
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

QUICK = bool(os.environ.get("QUICK"))
SEEDS_B1 = [1, 2] if QUICK else list(range(1, 17))
SEEDS_B3 = [1] if QUICK else [1, 2, 3, 4, 5, 7, 8, 9, 11]
OUT = os.path.join(HERE, "seeds_results_quick.jsonl" if QUICK else "seeds_results.jsonl")
T_B1A, T_B1B = (30.0, 40.0) if QUICK else (300.0, 200.0)
T_REL, T_REC = (20.0, 30.0) if QUICK else (100.0, 200.0)
ALPHAS = [0.3] if QUICK else [0.1, 0.3, 0.5, 0.8]
GBARS = [0.0, 0.3, 1.0] if QUICK else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
GBND = [0.0, 0.2, 1.0] if QUICK else [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


def reference(seed):
    p = mm.MinimalParams(N=60, seed=seed); net = mm.MinimalNetwork(p); net.run(300.0, record_every=300.0)
    return net.V.copy(), net.m.copy(), net.G.copy(), net.n.copy(), net.neighbors


def domains(V):
    sg = (V > 0).astype(int); out = []; start = 0
    for i in range(1, len(sg) + 1):
        if i == len(sg) or sg[i] != sg[start]:
            out.append((start, i - 1, int(sg[start]))); start = i
    return out


def centred_block(dom, K):
    a, b, _ = dom; c = (a + b) // 2; h = (K - 1) // 2
    return list(range(c - h, c + h + 1))


def frozen_run(seed_ref, alpha, V0, T, gbar=None, G_override=None):
    V_ref, m_ref, G_ref, n_ref, nbrs = seed_ref
    net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=1, alpha=alpha))
    if G_override is not None: G = G_override.copy()
    else:
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


def b1(seed):
    ref = reference(seed); V_ref, m_ref, G_ref, n_ref, nbrs = ref; dm = domains(V_ref)
    big = max(dm, key=lambda d: d[1] - d[0]); recs = []
    # B1a
    res = {}
    for K in (1, 3, 7):
        if (big[1] - big[0] + 1) < K + 4: continue
        om = centred_block(big, K); lo, hi = om[0], om[-1]
        bnd = [(lo - 1, lo), (hi, hi + 1)]; intl = [(i, i + 1) for i in range(lo, hi)]
        Es = []
        for gb in GBND:
            Vg, Gg = V_ref.copy(), G_ref.copy()
            for i in om: Vg[i] = -Vg[i]
            for (i, j) in bnd: Gg[i, j] = Gg[j, i] = gb
            for (i, j) in intl: Gg[i, j] = Gg[j, i] = 0.5
            Vend = frozen_run(ref, 0.30, Vg, T_B1A, G_override=Gg)
            Es.append(float(np.mean(np.abs(Vend[om] - V_ref[om]))))
        idx = [k for k, e in enumerate(Es) if e < 0.3]
        res[K] = dict(E=Es, first_recovered=GBND[idx[0]] if idx else None, domain=[big[0], big[1], big[2]])
    recs.append(dict(task="B1a", seed=seed, res=res))
    # B1b
    out = {}
    for name, pol in (("V-to-V+", 0), ("V+to-V-", 1)):
        cand = [d for d in dm if d[2] == pol and d[1] - d[0] + 1 >= 5]
        if not cand: out[name] = None; continue
        d = max(cand, key=lambda x: x[1] - x[0]); om = centred_block(d, 3); lo, hi = om[0], om[-1]
        bnd = [(lo - 1, lo), (hi, hi + 1)]; intl = [(lo, lo + 1), (lo + 1, hi)]
        net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=seed))
        Vg, Gg = V_ref.copy(), G_ref.copy()
        for i in om: Vg[i] = -Vg[i]
        for (i, j) in bnd + intl: Gg[i, j] = Gg[j, i] = 0.5
        net.V, net.m, net.G, net.n = Vg, m_ref.copy(), Gg, n_ref.copy(); net.time = 0.0; net._next_n_update = 1e9
        phi = lambda m: mm.phi_func(m, net.p.V_minus, net.p.V_plus, net.p.k_phi)
        ph0 = float(np.mean(phi(net.m[om])))
        for _ in range(int(T_B1B / net.p.dt)): net._rk4_step()
        out[name] = dict(domain=[d[0], d[1]], Gint=float(np.mean([net.G[i, j] for (i, j) in intl])),
                         Gbnd=float(np.mean([net.G[i, j] for (i, j) in bnd])), phi0=ph0, phi1=float(np.mean(phi(net.m[om]))),
                         still_inverted=bool(np.all(np.sign(net.V[om]) != np.sign(V_ref[om]))))
    recs.append(dict(task="B1b", seed=seed, res=out))
    return recs


def b3(seed):
    ref = reference(seed); V_ref = ref[0]; big = max(domains(V_ref), key=lambda d: d[1] - d[0]); res = {}
    for K in (1, 3, 7):
        if (big[1] - big[0] + 1) < K + 4: continue
        om = centred_block(big, K); S = []; ok = []
        for a in ALPHAS:
            row = []; okrow = []
            for g in GBARS:
                Vs = frozen_run(ref, a, V_ref, T_REL, gbar=g); okrow.append(bool(np.all(np.sign(Vs) == np.sign(V_ref))))
                V0 = Vs.copy(); V0[om] = -Vs[om]
                row.append(float(np.mean(np.sign(frozen_run(ref, a, V0, T_REC, gbar=g)[om]) != np.sign(Vs[om]))))
            S.append(row); ok.append(okrow)
        res[K] = dict(S=S, pattern_ok=ok, domain=[big[0], big[1], big[2]])
    return [dict(task="B3", seed=seed, alphas=ALPHAS, gbars=GBARS, res=res)]


def job(args):
    fn = {"B1": b1, "B3": b3}[args[0]]
    recs = fn(args[1])
    with open(OUT, "a") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    return args


if __name__ == "__main__":
    t0 = time.time()
    if os.path.exists(OUT): os.remove(OUT)
    tasks = [("B3", s) for s in SEEDS_B3] + [("B1", s) for s in SEEDS_B1]
    with Pool(4) as pool:
        for a in pool.imap_unordered(job, tasks): print(f"[{time.time()-t0:5.0f}s] done {a}", flush=True)
    print("results written to", OUT)
