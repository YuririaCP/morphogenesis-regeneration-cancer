"""
sim_blockade_ablation.py
========================
Which slow variable carries the effect of a transient transmission blockade (Prediction 3)?
Same protocol as sim_blockade.py (depolarized block of 3 cells inverted, transmission through its incident
edges multiplied by EPS_BLOCK for a duration D, then released), orientation V+ to V- only, with the
following variants applied DURING the blockade window only:
  nominal   : all layers evolve
  m frozen  : dm = 0 (memory cannot shift while the block is blocked)
  G frozen  : dG = 0 (the stored conductances cannot decay while the block is blocked)
Usage: python3 sim_blockade_ablation.py          QUICK=1 for a smoke test
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm
from sim_blockade import reference, domains, centred_block, EPS_BLOCK

QUICK = bool(os.environ.get("QUICK"))
SEEDS = [1, 4] if QUICK else [1, 2, 4, 9, 13, 15]
DS = [40.0, 100.0] if QUICK else [20.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
T_AFTER = 60.0 if QUICK else 300.0
OUT = os.path.join(HERE, "blockade_ablation_quick.jsonl" if QUICK else "blockade_ablation_results.jsonl")


def one_run(args):
    seed, variant, D = args
    V_ref, m_ref, G_ref, n_ref = reference(seed)
    cand = [d for d in domains(V_ref) if d[2] == 1 and d[1] - d[0] + 1 >= 5]
    d = max(cand, key=lambda x: x[1] - x[0]); om = centred_block(d, 3)
    edges = [(i, j) for (i, j) in zip(range(59), range(1, 60)) if (i in om or j in om)]
    scale = np.ones((60, 60))
    for (i, j) in edges: scale[i, j] = scale[j, i] = EPS_BLOCK
    net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=seed))
    V0 = V_ref.copy()
    for i in om: V0[i] = -V0[i]
    net.V, net.m, net.G, net.n = V0, m_ref.copy(), G_ref.copy(), n_ref.copy()
    net.time = 0.0; net._next_n_update = 1e9
    orig = net._derivatives
    def blocked(V, m, G, n):
        dVe, dme, _ = orig(V, m, G * scale, n)
        _, _, dGt = orig(V, m, G, n)
        if variant == "m frozen": dme = np.zeros_like(dme)
        if variant == "G frozen": dGt = np.zeros_like(dGt)
        return dVe, dme, dGt
    net._derivatives = blocked
    for _ in range(int(round(D / net.p.dt))): net._rk4_step()
    net._derivatives = orig
    for _ in range(int(round(T_AFTER / net.p.dt))): net._rk4_step()
    same = int(np.sum(np.sign(V_ref[om]) == np.sign(net.V[om])))
    return dict(seed=seed, variant=variant, D=D, outcome="recovered" if same == 3 else ("persists" if same == 0 else "mixed"))


if __name__ == "__main__":
    jobs = [(s, v, D) for s in SEEDS for v in ("nominal", "m frozen", "G frozen") for D in DS]
    t0 = time.time()
    with Pool() as pool, open(OUT, "w") as f:
        for k, r in enumerate(pool.imap_unordered(one_run, jobs)):
            f.write(json.dumps(r) + "\n"); f.flush()
    print("done", len(jobs), "runs in", round(time.time() - t0), "s ->", OUT)
