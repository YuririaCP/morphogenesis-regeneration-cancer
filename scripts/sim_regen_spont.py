"""
sim_regen_spont.py
==================
Regeneration from the spontaneously formed pattern (the same pattern as the morphogenesis and
cancer protocols), instead of a hand-initialized left/right state.

For each seed, the pattern is formed for T1 = 300 under the full dynamics. A block of K cells
interior to one domain is then lesioned (n_i = 0, V_i = m_i = 0, incident conductances zeroed) and the
system is followed for T3 = 100.  Outcomes: occupancy of the lesioned cells, correct sign
(a vacant cell counts as a failure), mean memory in the lesion, recovery time (time of the occupancy update, every 1.0,
that completes the repopulation; the first update occurs at t = 0), and the fraction of the whole tissue whose sign is unchanged.

Experiments
  A: seed 7, every domain, K in {1, 3} (K <= size - 2)          -> table_A
  B: ablations at lesion time on seed 7, first 5-cell depolarized domain, K = 3
  C: all 16 seeds, every domain with size >= K + 2, K in {1,3,5,7}

Usage: python3 sim_regen_spont.py            (about 10 minutes)
       QUICK=1 python3 sim_regen_spont.py    (smoke test)
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

QUICK = bool(os.environ.get("QUICK"))
T1, T3 = (60.0, 20.0) if QUICK else (300.0, 100.0)
SEEDS = range(1, 4) if QUICK else range(1, 17)

VARIANTS = {
    "nominal":       {},
    "kappa0":        {"kappa": 0.0},
    "alpha0":        {"alpha": 0.0},
    "alpha0_kappa0": {"alpha": 0.0, "kappa": 0.0},
    "frozen_memory": {"alpha": 0.0, "kappa": 0.0, "lam_m": 0.0},
}


def domains(V):
    sg = (V > 0).astype(int); out = []; s = 0
    for i in range(1, len(sg) + 1):
        if i == len(sg) or sg[i] != sg[s]:
            out.append((s, i - 1, int(sg[s]))); s = i
    return out


_CACHE = {}
def pattern(seed):
    if seed not in _CACHE:
        p = mm.MinimalParams(N=60, seed=seed); net = mm.MinimalNetwork(p); net.run(T1, record_every=T1)
        _CACHE[seed] = (net.V.copy(), net.m.copy(), net.G.copy(), net.n.copy())
    return _CACHE[seed]


def lesion_run(args):
    seed, lo, hi, variant = args
    V0, m0, G0, n0 = pattern(seed)
    p = mm.MinimalParams(N=60, seed=seed)
    for k, v in VARIANTS[variant].items(): setattr(p, k, v)
    net = mm.MinimalNetwork(p)
    net.V, net.m, net.G, net.n = V0.copy(), m0.copy(), G0.copy(), n0.copy(); net.time = 0.0
    net._next_n_update = 0.0
    net.impose_lesion(lo, hi)
    t_rec = None
    for t in range(1, int(T3) + 1):
        net.run(1.0, record_every=1.0)
        if t_rec is None and np.all(net.n[lo:hi] == 1): t_rec = t - 1   # time of the occupancy update (every 1.0) that completed the repopulation
    occ = float(net.n[lo:hi].mean())
    correct = float(np.mean([(net.n[i] == 1) and ((net.V[i] > 0) == (V0[i] > 0)) for i in range(lo, hi)]))
    same = float(np.mean((net.V > 0) == (V0 > 0)))
    return dict(seed=seed, lo=lo, hi=hi, K=hi - lo, variant=variant, occ=occ, correct=correct,
                m_mean=float(net.m[lo:hi].mean()), t_rec=t_rec, tissue_same=same)


def tasks_for(seed, Ks, variants=("nominal",)):
    V0 = pattern(seed)[0]; out = []
    for d in domains(V0):
        size = d[1] - d[0] + 1
        for K in Ks:
            if size < K + 2: continue
            c = (d[0] + d[1] + 1) // 2; lo = c - K // 2 if K % 2 else c - K // 2
            lo = max(d[0] + 1, min(lo, d[1] + 1 - K - 1 + 1)); hi = lo + K
            if lo < d[0] + 1 or hi > d[1]: continue
            for v in variants: out.append((seed, lo, hi, v, d[2], size))
    return out


def run_tasks(tasks):
    meta = {(t[0], t[1], t[2], t[3]): (t[4], t[5]) for t in tasks}
    with Pool(4) as pool: res = pool.map(lesion_run, [t[:4] for t in tasks])
    for r in res: r["pol"], r["size"] = meta[(r["seed"], r["lo"], r["hi"], r["variant"])]
    return res


if __name__ == "__main__":
    t0 = time.time(); out = {}
    # A: seed 7 (the pattern of the morphogenesis and cancer protocols)
    V7 = pattern(7)[0]
    print("seed 7 domains (start, end, pol):", domains(V7), flush=True)
    A = run_tasks(tasks_for(7, [1, 3]))
    out["A"] = A
    print(f"[{time.time()-t0:.0f}s] A: {len(A)} runs", flush=True)
    for r in A: print("  A", r["pol"], r["size"], "K", r["K"], "cells", r["lo"], r["hi"], "occ %.2f correct %.2f t_rec %s same %.2f" % (r["occ"], r["correct"], r["t_rec"], r["tissue_same"]), flush=True)
    # B: ablations, seed 7, first 5-cell depolarized domain
    dep5 = [d for d in domains(V7) if d[2] == 1 and d[1] - d[0] + 1 == 5]
    if dep5:
        d = dep5[0]; lo = d[0] + 1; hi = lo + 3
        B = run_tasks([(7, lo, hi, v, 1, 5) for v in VARIANTS])
        out["B"] = B
        for r in B: print("  B", r["variant"], "occ %.2f correct %.2f m_mean %.3f t_rec %s" % (r["occ"], r["correct"], r["m_mean"], r["t_rec"]), flush=True)
    # C: sweep over seeds
    tc = []
    for s in SEEDS: tc += tasks_for(s, [1, 3, 5, 7])
    C = run_tasks(tc); out["C"] = C
    print(f"[{time.time()-t0:.0f}s] C: {len(C)} runs", flush=True)
    for pol, name in ((1, "depolarized"), (0, "hyperpolarized")):
        for K in (1, 3, 5, 7):
            rr = [r for r in C if r["pol"] == pol and r["K"] == K]
            if rr: print(f"  C {name:15s} K={K}: n={len(rr):3d} full refill {sum(r['occ']==1.0 for r in rr):3d}/{len(rr):3d}  "
                         f"correct {np.mean([r['correct'] for r in rr]):.2f}  tissue unchanged {np.mean([r['tissue_same'] for r in rr]):.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "regen_spont_results%s.json" % ("_quick" if QUICK else "")), "w"))
    print("saved; total %.0fs" % (time.time() - t0))
