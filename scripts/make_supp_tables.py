"""
make_supp_tables.py
===================
Writes the LaTeX tables of the Supplementary Material from the stored results of sim_seeds.py
(seeds_results.jsonl) and sim_seeds_env.py (seeds_env_results.json).

Usage: python3 make_supp_tables.py
"""
import json, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "seeds_results.jsonl"))]
ENV = json.load(open(os.path.join(HERE, "seeds_env_results.json")))
GBND = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


def fmt(v):
    return r"$>1$" if v is None else f"{v:g}"


# ---------------- B1a: recovery threshold per seed -----------------------------------------------
rows = []
for r in sorted([r for r in R if r["task"] == "B1a"], key=lambda r: r["seed"]):
    d = r["res"]["1"]["domain"]
    pol = "$V_+$" if d[2] == 1 else "$V_-$"
    rows.append(f"{r['seed']} & {d[0]}--{d[1]} ({pol}) & " + " & ".join(
        fmt(r["res"][k]["first_recovered"]) if k in r["res"] else "--" for k in ("1", "3", "7")) + r" \\")
open(os.path.join(HERE, "supp_tab_b1a.tex"), "w").write(
    "\\begin{tabular}{llccc}\n\\toprule\nSeed & Host domain & $|\\Omega|=1$ & $|\\Omega|=3$ & $|\\Omega|=7$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- B1b: adaptive isolation per seed ---------------------------------------------------
rows = []
for r in sorted([r for r in R if r["task"] == "B1b"], key=lambda r: r["seed"]):
    cells = []
    for name in ("V-to-V+", "V+to-V-"):
        x = r["res"].get(name)
        if x is None: cells.append("-- & -- & --")
        else:
            d = x["domain"]; size = d[1] - d[0] + 1
            cells.append(f"{size} & {'persists' if x['still_inverted'] else 'recovers'} & ${x['phi0']:+.2f}\\to{x['phi1']:+.2f}$")
    rows.append(f"{r['seed']} & " + " & ".join(cells) + r" \\")
open(os.path.join(HERE, "supp_tab_b1b.tex"), "w").write(
    "\\begin{tabular}{rrlcrlc}\n\\toprule\n & \\multicolumn{3}{c}{$V_-\\to V_+$ (hyperpolarised host)} & \\multicolumn{3}{c}{$V_+\\to V_-$ (depolarised host)} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nSeed & host size & outcome & $\\overline{\\varphi(m)}$ & host size & outcome & $\\overline{\\varphi(m)}$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- environment control ------------------------------------------------------------------
lines = []
for env, label in (("native", "block and its boundary only"), ("domain", "every edge of the host domain"), ("uniform", "every edge of the tissue")):
    cells = []
    for pol in (0, 1):
        rr = [x for x in ENV if x["env"] == env and x["pol"] == pol]
        cells.append(f"{sum(x['inverted'] for x in rr)}/{len(rr)}")
    lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
open(os.path.join(HERE, "supp_tab_env.tex"), "w").write(
    "\\begin{tabular}{lcc}\n\\toprule\nInitial conductance $0.5$ on & $V_-\\to V_+$ persists & $V_+\\to V_-$ persists \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- B3 per seed ------------------------------------------------------------------------------
b3 = sorted([r for r in R if r["task"] == "B3"], key=lambda r: r["seed"])
al, gb = b3[0]["alphas"], b3[0]["gbars"]
def first(row):
    idx = [k for k, x in enumerate(row) if x < 0.5]
    return None if not idx else gb[idx[0]]
rows = []
for r in b3:
    d = r["res"]["1"]["domain"]
    cells = []
    for K in ("1", "3", "7"):
        S = np.array(r["res"][K]["S"])
        cells.append(" / ".join("--" if first(S[i]) is None else f"{first(S[i]):g}" for i in range(len(al))))
    rows.append(f"{r['seed']} & {d[0]}--{d[1]} & " + " & ".join(cells) + r" \\")
open(os.path.join(HERE, "supp_tab_b3.tex"), "w").write(
    "\\begin{tabular}{llccc}\n\\toprule\nSeed & Host domain & $|\\Omega|=1$ & $|\\Omega|=3$ & $|\\Omega|=7$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("tables written")
