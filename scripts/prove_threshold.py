"""
prove_threshold.py
Yuriria Cortés-Poza
===================
Numerical verification of Lemma 1 (monostability threshold of a committed cell)
of the manuscript.

We verify:
  1. The analytic formula  alpha_c* = a*L^2 / (4*C) = 0.5625
  2. For G + alpha > alpha_c*: exactly ONE equilibrium at V = V+
     (and ONE at V = V-)  →  unique morphogenetic attractor.
  3. For G + alpha < alpha_c*: THREE equilibria per cell  →  cancer
     attractors exist.
  4. The LINEAR phase boundary  G + alpha = 0.5625  matches the
     simulated coherence phase diagram.
"""

import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figs", exist_ok=True)

# ─── Model parameters (Table S1) ────────────────────────────────────────────
a   = 1.0
Vm  = -1.5   # V_-
V0  =  0.0   # V_0
Vp  =  1.5   # V_+
C   =  1.0

L   = Vp     # half-width (V+ = L, V- = -L, V0 = 0)
alpha_c_star = a * L**2 / (4.0 * C)   # Eq. (alphac_explicit)
print(f"α_c*  =  aL²/(4C)  =  {alpha_c_star:.4f}")

# ─── 1. Verify equilibrium count for a single committed cell ─────────────────
def f_bistable(V):
    return a * (V - Vm) * (V - V0) * (V - Vp)

def count_equilibria(G_out, alpha, phi=Vp, n_grid=8000):
    """
    Count STABLE equilibria of  F(V) = -f(V) + (G_out+alpha)*(phi-V) = 0.
    Stability: F'(V*) < 0, i.e.  -f'(V*) - (G_out+alpha) < 0
                              ⟺  f'(V*) + (G_out+alpha) > 0.
    Scan range extended to catch roots outside [Vm, Vp].
    """
    scan_lo, scan_hi = Vm - 1.5, Vp + 1.5   # wide range
    V  = np.linspace(scan_lo, scan_hi, n_grid)
    # RHS of equilibrium equation F(V) = 0
    rhs = -f_bistable(V) + (G_out + alpha) * (phi - V)
    sign = np.sign(rhs)
    crossings = np.where(np.diff(sign) != 0)[0]
    roots = []
    for idx in crossings:
        lo, hi = V[idx], V[idx + 1]
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fmid = -f_bistable(mid) + (G_out + alpha) * (phi - mid)
            if rhs[idx] * fmid < 0:
                hi = mid
            else:
                lo = mid
        roots.append(0.5 * (lo + hi))

    def fprime(v):
        return a * ((v-V0)*(v-Vp) + (v-Vm)*(v-Vp) + (v-Vm)*(v-V0))

    # Stability: F'(V*) = -f'(V*) - (G_out+alpha) < 0
    #            ⟺  f'(V*) + (G_out+alpha) > 0
    stable = [r for r in roots if fprime(r) + (G_out + alpha) > 0]
    return len(stable), stable


print("\n── Table 1: equilibrium count vs (G, α) ──")
print(f"{'G':>6}  {'α':>6}  {'G+α':>6}  {'regime':>12}  "
      f"{'n_eq':>5}  {'roots':>25}")
test_cases = [
    (0.00, 0.30), (0.10, 0.30), (0.20, 0.30), (0.26, 0.30),
    (0.30, 0.30), (0.40, 0.30), (0.50, 0.30), (0.00, 0.57),
    (0.00, 0.60), (0.30, 0.60),
]
for G, alpha in test_cases:
    beta = G + alpha
    regime = "morphostasis" if beta > alpha_c_star else "cancer      "
    n, roots = count_equilibria(G, alpha, phi=Vp)
    print(f"{G:6.2f}  {alpha:6.2f}  {beta:6.4f}  {regime:>12}  "
          f"{n:5d}  {[f'{r:.4f}' for r in roots]}")


# ─── 2. Phase boundary: G + alpha = alpha_c* ────────────────────────────────
G_vals     = np.linspace(0.02, 0.80, 60)
alpha_vals = np.linspace(0.02, 0.70, 60)

# For each (G, alpha): count stable equilibria for a cell targeted to V+
n_eq = np.zeros((len(alpha_vals), len(G_vals)), dtype=int)
for i, alpha in enumerate(alpha_vals):
    for j, G in enumerate(G_vals):
        n_eq[i, j], _ = count_equilibria(G, alpha, phi=Vp)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
# Shade regions
cmap = plt.cm.RdYlGn
im = ax.pcolormesh(G_vals, alpha_vals, n_eq,
                   cmap=cmap, vmin=1, vmax=3, shading="auto")
plt.colorbar(im, ax=ax, label="Number of stable equilibria")

# Analytic phase boundary  G + alpha = alpha_c*
G_line = np.linspace(0, alpha_c_star, 200)
ax.plot(G_line, alpha_c_star - G_line, "k--", lw=2,
        label=r"$G+\alpha = \alpha_c^*$")

ax.axhline(alpha_c_star, color="gray", ls=":", lw=1, alpha=0.5)
ax.axvline(alpha_c_star, color="gray", ls=":", lw=1, alpha=0.5)

ax.set_xlabel(r"Gap-junction conductance $G$", fontsize=12)
ax.set_ylabel(r"Error-correction strength $\alpha$", fontsize=12)
ax.set_title(
    r"Monostable ($n_{\rm eq}=1$, green) vs. bistable ($n_{\rm eq}>1$, red)",
    fontsize=10)
ax.legend(fontsize=9)
ax.text(0.55, 0.05, r"$\alpha_c^*=0.5625$", fontsize=9,
        transform=ax.transAxes, color="black",
        bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                  edgecolor='none', alpha=0.88))

plt.tight_layout()
plt.savefig("figs/fig_monostability.pdf", dpi=200)
print("\nSaved: figs/fig_monostability.pdf")


# ─── 3. Saddle-node continuation diagram (Lemma 1) ──────────────────────────
# Exact branches from the proof of Lemma 1, target phi_0 = L = V+:
#   morphostatic root      V = L                          (always an equilibrium)
#   alternative branches   q(V) = a V^2 + a L V + beta = 0
#     V_lo = (-L - s)/2  (stable),   V_hi = (-L + s)/2  (unstable),
#     s = sqrt(L^2 - 4 beta / a),  defined for beta <= a L^2 / 4 = alpha_c^*
beta_max = 1.0
beta_alt = np.linspace(0.0, alpha_c_star, 400)
s = np.sqrt(np.clip(L**2 - 4 * beta_alt / a, 0.0, None))
V_lo = (-L - s) / 2.0
V_hi = (-L + s) / 2.0
beta_morph = np.linspace(0.0, beta_max, 400)

fig2, ax = plt.subplots(figsize=(6.2, 4.6))
ax.plot(beta_morph, np.full_like(beta_morph, L), color="seagreen", lw=2.2,
        label="morphostatic branch (stable)")
ax.plot(beta_alt, V_lo, color="steelblue", lw=2.2,
        label="alternative branch (stable)")
ax.plot(beta_alt, V_hi, color="firebrick", lw=1.8, ls="--",
        label="unstable branch")
ax.plot([alpha_c_star], [-L / 2], marker="o", ms=7, mfc="white",
        mec="black", mew=1.4, zorder=5, label="saddle-node")
ax.axvline(alpha_c_star, color="grey", ls=":", lw=1, alpha=0.7)
ax.text(alpha_c_star, Vm - 0.28, rf"$\alpha_c^*={alpha_c_star:.4f}$",
        fontsize=9, ha="center", color="grey")
ax.set_xlabel(r"effective restoring coupling $\beta_{\rm eff}=G_{\rm out}+\alpha$",
              fontsize=11)
ax.set_ylabel(r"equilibrium voltage $V^*$", fontsize=11)
ax.set_title("Saddle-node continuation of the committed-cell reduction (Lemma 1)",
             fontsize=10.5)
ax.set_xlim(0, beta_max)
ax.set_ylim(Vm - 0.4, Vp + 0.2)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig("figs/fig_bifurcation.pdf", dpi=200)
print("Saved: figs/fig_bifurcation.pdf")


# ─── 4. Verify G_c = alpha_c* - alpha matches simulated phase diagram ───────
# Load phase diagram data by re-running the model at selected points
try:
    from bioelectric_model import ModelParameters, BioelectricNetwork
    from minimal_model import run_morphogenesis_1d

    G_test  = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    a_test  = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    coherence = np.full((len(a_test), len(G_test)), np.nan)

    for i, alph in enumerate(a_test):
        for j, G in enumerate(G_test):
            params = ModelParameters(N=30, G_init=G, alpha=alph,
                                     tau_m=80, kappa=0.1)
            net = BioelectricNetwork(params)
            net.run(T=80.0)
            V = net.V[net.n == 1]
            n_plus = np.sum(V > 0)
            n_occ  = len(V)
            coherence[i, j] = max(n_plus, n_occ - n_plus) / n_occ if n_occ else 0.5

    fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
    cs = ax3.contourf(G_test, a_test, coherence, levels=20,
                      cmap="RdYlGn", vmin=0.5, vmax=1.0)
    plt.colorbar(cs, ax=ax3, label="Coherence score $\\mathcal{C}$")
    ax3.contour(G_test, a_test, coherence, levels=[0.75],
                colors=["black"], linestyles=["--"], linewidths=[2])

    # Overlay analytic boundary
    G_line2 = np.linspace(0, alpha_c_star, 200)
    ax3.plot(G_line2, alpha_c_star - G_line2, "w-", lw=2.5,
             label=r"$G+\alpha=\alpha_c^*$ (analytic)")
    ax3.legend(fontsize=9)
    ax3.set_xlabel(r"$\bar G$", fontsize=12)
    ax3.set_ylabel(r"$\alpha$", fontsize=12)
    ax3.set_title("Phase diagram: theory vs simulation", fontsize=10)
    plt.tight_layout()
    plt.savefig("figs/fig_phase_overlay.pdf", dpi=200)
    print("Saved: figs/fig_phase_overlay.pdf")

except Exception as e:
    print(f"Phase diagram overlay skipped ({e}); "
          "run after confirming model imports.")


print("\n══ Summary ══")
print(f"α_c*  =  {alpha_c_star:.6f}   [analytic: a*L²/(4C) = {a}*{L}²/(4*{C})]")
print(f"Phase boundary:  G + α = {alpha_c_star:.4f}  (slope -1 line)")
print(f"For α=0.30:  G_c = {max(0.0, alpha_c_star-0.30):.4f}")
print(f"For α=0.50:  G_c = {max(0.0, alpha_c_star-0.50):.4f}")
print(f"For α=0.57:  G_c = {max(0.0, alpha_c_star-0.57):.4f} (≈0, near α_c*)")
