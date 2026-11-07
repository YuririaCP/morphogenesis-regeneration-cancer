"""
sim_seeds_env.py
================
Control for the adaptive-isolation experiment across seeds: inversion of a 3-cell block in the largest
hyperpolarized (V-to-V+) or depolarized (V+to-V-) domain with >= 7 cells, full adaptive dynamics for T = 200,
with initial conductance 0.5 on (native) the block and its boundary only, (domain) every edge of the host domain,
or (uniform) every edge of the tissue. Results: seeds_env_results.json. Usage: python3 sim_seeds_env.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import minimal_model as mm
import sim_seeds as S
from multiprocessing import Pool

T = 200.0

def run(seed, pol, env):
    ref = S.reference(seed); V_ref, m_ref, G_ref, n_ref, nbrs = ref; dm = S.domains(V_ref)
    cand = [d for d in dm if d[2] == pol and d[1] - d[0] + 1 >= 7]
    if not cand: return None
    d = max(cand, key=lambda x: x[1] - x[0]); om = S.centred_block(d, 3); lo, hi = om[0], om[-1]
    Gg = G_ref.copy()
    if env == 'native':          # only Omega and its boundary edges set to 0.5 (as in the manuscript)
        edges = [(lo - 1, lo), (lo, lo + 1), (lo + 1, hi), (hi, hi + 1)]
    elif env == 'domain':        # every edge inside the host domain set to 0.5
        edges = [(i, i + 1) for i in range(d[0], d[1])]
    else:                        # every edge of the tissue set to 0.5
        edges = [(i, i + 1) for i in range(59)]
    for (i, j) in edges: Gg[i, j] = Gg[j, i] = 0.5
    Vg = V_ref.copy()
    for i in om: Vg[i] = -Vg[i]
    net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=seed))
    net.V, net.m, net.G, net.n = Vg, m_ref.copy(), Gg, n_ref.copy(); net.time = 0.0; net._next_n_update = 1e9
    phi = lambda m: mm.phi_func(m, net.p.V_minus, net.p.V_plus, net.p.k_phi)
    for _ in range(int(T / net.p.dt)): net._rk4_step()
    intl = [(lo, lo + 1), (lo + 1, hi)]; bnd = [(lo - 1, lo), (hi, hi + 1)]
    return dict(seed=seed, pol=pol, env=env, size=d[1] - d[0] + 1,
                inverted=bool(np.all(np.sign(net.V[om]) != np.sign(V_ref[om]))),
                Gint=float(np.mean([net.G[i, j] for (i, j) in intl])), Gbnd=float(np.mean([net.G[i, j] for (i, j) in bnd])),
                phi1=float(np.mean(phi(net.m[om]))))

def job(a): return run(*a)

if __name__ == "__main__":
    tasks = [(s, pol, env) for s in range(1, 17) for pol in (0, 1) for env in ('native', 'domain', 'uniform')]
    with Pool(4) as p: res = [r for r in p.map(job, tasks) if r]
    for env in ('native', 'domain', 'uniform'):
        for pol, name in ((0, 'V-to-V+ (host hyperpolarized)'), (1, 'V+to-V- (host depolarized)')):
            rr = [r for r in res if r['env'] == env and r['pol'] == pol]
            inv = sum(r['inverted'] for r in rr)
            print(f"{env:8s} {name:30s} n={len(rr):2d}  stays inverted: {inv:2d}/{len(rr)}  "
                  f"mean Gint={np.mean([r['Gint'] for r in rr]):.2f} mean Gbnd={np.mean([r['Gbnd'] for r in rr]):.3f}  phi1 mean={np.mean([r['phi1'] for r in rr]):+.2f}")
    json.dump(res, open(os.path.join(HERE, "seeds_env_results.json"), "w"))
