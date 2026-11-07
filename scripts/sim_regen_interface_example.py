"""
sim_regen_interface_example.py
==============================
Scope control for the regeneration experiments: one lesion across one polarity interface of the pinned
seed-7 pattern (N = 60, T1 = 300), evolved under the full adaptive dynamics with the occupancy rule.
Cells 17-22 (three on each side of the interface at 19.5) are removed; the script reports, for each side,
the fraction of cells that are occupied again and that carry the sign they had before the lesion, and the
position of the interface before and after.

This single example only delimits the regeneration regime considered in the main text; it is not a
systematic study of interfaces.

Usage: python3 sim_regen_interface_example.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import minimal_model as mm

T_POST = 100.0
LESION, PLUS, MINUS = list(range(17, 23)), [17, 18, 19], [20, 21, 22]

p = mm.MinimalParams(N=60, seed=7)
ref = mm.MinimalNetwork(p); ref.run(300.0, record_every=300.0)
V_ref, m_ref, G_ref, n_ref = ref.V.copy(), ref.m.copy(), ref.G.copy(), ref.n.copy()

net = mm.MinimalNetwork(mm.MinimalParams(N=60, seed=7))
net.V, net.m, net.G, net.n = V_ref.copy(), m_ref.copy(), G_ref.copy(), n_ref.copy(); net.time = 300.0
for i in LESION:
    net.n[i] = 0; net.V[i] = 0.0; net.m[i] = 0.0
    for j in net.neighbors[i]: net.G[i, j] = net.G[j, i] = 0.0
net._next_n_update = net.time + p.dt_n
t = 300.0
for _ in range(int(round(T_POST / p.dt))):
    if t >= net._next_n_update: net._update_occupancy(); net._next_n_update += p.dt_n
    net._rk4_step(); t += p.dt


def interface(V, n):
    """position of the sign change nearest to 19.5 among occupied cells"""
    idx = [i for i in range(60) if n[i] == 1]; pos = []
    for a, b in zip(idx[:-1], idx[1:]):
        if (V[a] > 0) != (V[b] > 0): pos.append((a + b) / 2)
    return min(pos, key=lambda x: abs(x - 19.5)) if pos else float('nan')


sg_ref = V_ref > 0
for name, side in (("depolarized side", PLUS), ("hyperpolarized side", MINUS)):
    occ = net.n[side].astype(bool); same = occ & ((net.V[side] > 0) == sg_ref[side])
    print(f"{name:20s}: occupied {occ.mean():.2f}, occupied with the original sign {same.mean():.2f}")
print(f"interface before: {interface(V_ref, n_ref):.1f}   after: {interface(net.V, net.n):.1f}")
