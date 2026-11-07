"""
2D Bioelectric Field Model – Simulation
Cortés-Poza, "Morphogenesis, Regeneration, and Cancer-Like Autonomy in a
Common Bioelectric Dynamical System"

Three protocols on a 20×20 grid:
  (A) Target convergence – random IC, G_ij=0 → prescribed pattern
  (B) Pattern recovery – morphostatic state → left half scrambled → pattern restored
  (C) Cancer-like autonomy – morphostatic state → local G loss → local alternative state

ODE system (per cell i, neighbours j ∈ N(i)):
  C  V̇_i  = −f(V_i) + α(φ(m_i)−V_i) + Σ_j G_ij(V_j−V_i)
  τ_G Ġ_ij = σ(V_i)σ(V_j)(1+γ m_i m_j) − λ_G G_ij
  τ_m ṁ_i  = −λ_m m_i + σ_h(V_i) + κ Σ_j G_ij(m_j−m_i)

f(V) = a V(V+L)(V−L)   [note: isolated cell eq. is C dV/dt = −f(V)]
"""

import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

# ── Parameters ─────────────────────────────────────────────────────────────
N    = 20           # grid side (N×N cells)
NC   = N * N

# bistable cubic  f(V) = a V(V+L)(V−L)
a    = 1.0
L    = 1.5
Vp   = L            # V_+
Vm   = -L           # V_−

# morphostasis threshold (symmetric cubic)
alpha_c = a * L**2 / 4   # = 0.5625

# model parameters
C      = 1.0
alpha  = 1.2           # > alpha_c  → morphostatic
gamma  = 2.0           # memory coupling bonus
lam_G  = 0.5           # G decay rate
lam_m  = 0.3           # memory decay rate
kappa  = 0.1           # memory diffusion
G0     = 1.0           # baseline gap-junction strength
tau_G  = 5.0
tau_m  = 50.0

# sigmoidal functions
k_sig  = 4.0
def sigma(V):
    """Gap-junction opening: high at V_+, low at V_−."""
    return 1.0 / (1.0 + np.exp(-k_sig * V))

def sigma_h(V):
    """Memory driving: high at V_+, low at V_−."""
    return 1.0 / (1.0 + np.exp(-k_sig * V))

# bistable force
def f_bist(V):
    return a * V * (V + L) * (V - L)

# morphogenetic target
def phi(m):
    """Interpolate: m=1 → V_+, m=0 → V_−."""
    return Vm + (Vp - Vm) * m

# ── Grid / adjacency ────────────────────────────────────────────────────────
def flat(r, c):   return r * N + c
def grid_rc(i):   return divmod(i, N)

def build_adjacency():
    """Returns list of (i,j) neighbour pairs for 4-connected grid."""
    pairs = []
    for r in range(N):
        for c in range(N):
            i = flat(r, c)
            if r + 1 < N: pairs.append((i, flat(r+1, c)))
            if c + 1 < N: pairs.append((i, flat(r, c+1)))
    return pairs

PAIRS = build_adjacency()
NPAIRS = len(PAIRS)

# pair → (i, j) arrays for vectorised ops
PI = np.array([p[0] for p in PAIRS], dtype=int)
PJ = np.array([p[1] for p in PAIRS], dtype=int)

def G_tot(G_pairs):
    """Total conductance for each cell: G_i^tot = Σ_j G_ij."""
    gt = np.zeros(NC)
    np.add.at(gt, PI, G_pairs)
    np.add.at(gt, PJ, G_pairs)
    return gt

def laplacian_V(V, G_pairs):
    """Σ_j G_ij (V_j − V_i) for each i."""
    lv = np.zeros(NC)
    diff = V[PJ] - V[PI]
    np.add.at(lv, PI,  G_pairs * diff)
    np.add.at(lv, PJ, -G_pairs * diff)
    return lv

def laplacian_m(m, G_pairs):
    """Σ_j G_ij (m_j − m_i) for each i."""
    lm = np.zeros(NC)
    diff = m[PJ] - m[PI]
    np.add.at(lm, PI,  G_pairs * diff)
    np.add.at(lm, PJ, -G_pairs * diff)
    return lm

# ── Morphological identity pattern ──────────────────────────────────────────
def stripe_identity(width=N//2):
    """Left half → identity 1 (target V_+), right half → identity 0 (target V_−)."""
    ident = np.zeros(NC)
    for r in range(N):
        for c in range(N):
            if c < width:
                ident[flat(r, c)] = 1.0
    return ident

IDENT = stripe_identity()   # fixed morphological identity

def target_phi(ident):
    return phi(ident)       # V_+ for id=1, V_− for id=0

# ── ODE right-hand side ─────────────────────────────────────────────────────
def rhs(t, y, G_fixed=None):
    """
    State vector: y = [V (NC), G_pairs (NPAIRS), m (NC)]
    If G_fixed is not None, gap junctions are clamped (cancer/ablation).
    """
    V       = y[:NC]
    G_pairs = y[NC:NC+NPAIRS] if G_fixed is None else G_fixed
    m       = y[NC+NPAIRS:]

    phi_i = target_phi(IDENT)   # morphogenetic targets (fixed identity)

    # --- V equation ---
    lv   = laplacian_V(V, G_pairs)
    dV   = (1.0/C) * (-f_bist(V) + alpha*(phi_i - V) + lv)

    # --- G equation ---
    if G_fixed is None:
        sig_i = sigma(V[PI]);  sig_j = sigma(V[PJ])
        mi    = m[PI];         mj    = m[PJ]
        dG = (1.0/tau_G) * (sig_i * sig_j * (1.0 + gamma*mi*mj) - lam_G*G_pairs)
    else:
        dG = np.zeros(NPAIRS)

    # --- m equation ---
    lm   = laplacian_m(m, G_pairs)
    dm   = (1.0/tau_m) * (lam_m*(sigma_h(V) - m) + kappa*lm)

    return np.concatenate([dV, dG, dm])

# ── Initial conditions ──────────────────────────────────────────────────────
def random_IC(seed=42):
    rng = np.random.default_rng(seed)
    V   = rng.uniform(Vm, Vp, NC)
    G   = rng.uniform(0.0, 0.2, NPAIRS)
    m   = rng.uniform(0.0, 1.0, NC)
    return np.concatenate([V, G, m])

def morphostatic_IC(seed=0):
    """Approximately converged morphostatic state."""
    rng = np.random.default_rng(seed)
    V   = target_phi(IDENT) + rng.normal(0, 0.05, NC)
    G   = G0 * np.ones(NPAIRS) + rng.normal(0, 0.02, NPAIRS)
    G   = np.clip(G, 0, None)
    m   = IDENT.copy() + rng.normal(0, 0.02, NC)
    m   = np.clip(m, 0, 1)
    return np.concatenate([V, G, m])

# ── Solvers for each protocol ───────────────────────────────────────────────
def solve(y0, t_span, t_eval, G_fixed=None, rtol=1e-4, atol=1e-6):
    sol = solve_ivp(
        lambda t, y: rhs(t, y, G_fixed=G_fixed),
        t_span, y0, method="RK45",
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=0.5
    )
    return sol

def extract(sol, t_idx):
    y = sol.y[:, t_idx]
    V = y[:NC].reshape(N, N)
    m = y[NC+NPAIRS:].reshape(N, N)
    return V, m

# ── Protocol A: Morphogenesis ────────────────────────────────────────────────
print("Protocol A: Morphogenesis …")
y0_A  = random_IC(seed=7)
tA    = np.linspace(0, 500, 300)
solA  = solve(y0_A, (0, 500), tA)

# ── Protocol B: Regeneration ─────────────────────────────────────────────────
print("Protocol B: Regeneration …")
# Start from morphostatic IC, ablate left half at t=200, run to t=600
y0_B  = morphostatic_IC()
tB1   = np.linspace(0, 200, 100)
solB1 = solve(y0_B, (0, 200), tB1)

# Ablation: reset V and m in the left half to random (injury)
y_abl = solB1.y[:, -1].copy()
rng_abl = np.random.default_rng(11)
for r in range(N):
    for c in range(N//2):
        i = flat(r, c)
        y_abl[i]           = rng_abl.uniform(-0.5, 0.5)      # V scrambled
        y_abl[NC+NPAIRS+i] = rng_abl.uniform(0.0, 0.5)       # m scrambled

tB2   = np.linspace(200, 700, 200)
solB2 = solve(y_abl, (200, 700), tB2)

# ── Protocol C: Cancer ───────────────────────────────────────────────────────
# Physics note: cancer requires alpha_eff = G_patch + alpha_C < alpha_c*.
# With alpha=1.2 (used in A & B) the memory term alone exceeds the threshold
# even at G=0, so no cancer attractor forms.  We use alpha_cancer=0.30, so
# beta_eff = G_lo + alpha_cancer = 0.015 + 0.30 = 0.315 < alpha_c* = 0.5625.
# This matches the 1D cancer protocol (fig_cancer).
alpha_cancer = 0.30   # < alpha_c* = 0.5625
G_lo         = 0.015  # residual conductance in cancer patch (matches 1D)

def rhs_cancer(t, y, cancer_pair_mask):
    """RHS with alpha_cancer and G fixed at G_lo in cancer-adjacent pairs."""
    V       = y[:NC]
    G_pairs = y[NC:NC+NPAIRS].copy()
    G_pairs[cancer_pair_mask] = G_lo     # enforce reduced conductance
    m       = y[NC+NPAIRS:]
    phi_i   = target_phi(IDENT)
    lv      = laplacian_V(V, G_pairs)
    dV      = (1.0/C) * (-f_bist(V) + alpha_cancer*(phi_i - V) + lv)
    sig_i = sigma(V[PI]); sig_j = sigma(V[PJ])
    mi = m[PI]; mj = m[PJ]
    dG = (1.0/tau_G) * (sig_i*sig_j*(1.0 + gamma*mi*mj) - lam_G*G_pairs)
    dG[cancer_pair_mask] = 0.0           # G fixed in cancer patch
    lm = laplacian_m(m, G_pairs)
    dm = (1.0/tau_m) * (lam_m*(sigma_h(V) - m) + kappa*lm)
    return np.concatenate([dV, dG, dm])

print("Protocol C: Cancer …")
# Phase-1: morphostatic warm-up with alpha_cancer
def solve_C(y0, t_span, t_eval, cancer_pair_mask=None):
    if cancer_pair_mask is None:
        return solve_ivp(
            lambda t, y: rhs_cancer(t, y, np.zeros(NPAIRS, dtype=bool)),
            t_span, y0, method="RK45",
            t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=0.5)
    return solve_ivp(
        lambda t, y: rhs_cancer(t, y, cancer_pair_mask),
        t_span, y0, method="RK45",
        t_eval=t_eval, rtol=1e-4, atol=1e-6, max_step=0.5)

y0_C  = morphostatic_IC(seed=3)
tC1   = np.linspace(0, 200, 100)
solC1 = solve_C(y0_C, (0, 200), tC1)   # no cancer mask yet

# Cancer trigger: 8×8 patch centred in the right half (identity=0 region)
cancer_cells = set(flat(r, c) for r in range(6, 14) for c in range(12, 20))
cancer_pair_mask = np.array(
    [PI[k] in cancer_cells or PJ[k] in cancer_cells for k in range(NPAIRS)],
    dtype=bool)

y_cancer = solC1.y[:, -1].copy()
# Reduce G to G_lo in cancer pairs
for k in range(NPAIRS):
    if cancer_pair_mask[k]:
        y_cancer[NC + k] = G_lo
# Invert voltages in cancer patch (bioelectric de-differentiation)
for i in cancer_cells:
    y_cancer[i] = -y_cancer[i]

tC2   = np.linspace(200, 700, 200)
solC2 = solve_C(y_cancer, (200, 700), tC2, cancer_pair_mask=cancer_pair_mask)

# ── Plotting ─────────────────────────────────────────────────────────────────
print("Plotting …")

cmap_V = "RdBu_r"
cmap_m = "viridis"
vmin_V, vmax_V = Vm, Vp
vmin_m, vmax_m = 0.0, 1.0

fig = plt.figure(figsize=(7.2, 4.3))
fig.patch.set_facecolor("white")
# columns: 4 voltage snapshots | V colourbar | spacer | memory | m colourbar
gs = gridspec.GridSpec(3, 8, figure=fig, width_ratios=[1, 1, 1, 1, 0.06, 0.42, 1.05, 0.06],
                       wspace=0.12, hspace=0.35, left=0.15, right=0.97, top=0.95, bottom=0.03)


def show(ax, data, cmap, vmin, vmax, title):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    return im


rows = []
# Row A - target convergence
snap_A = [0, len(tA)//5, 2*len(tA)//5, -1]
titles_A = ["t=0 (random)", "t=100", "t=200", "t=500"]
rows.append(([extract(solA, ti)[0] for ti in snap_A], titles_A, extract(solA, -1)[1], "memory, t=500"))
# Row B - pattern recovery
V0_B, _ = extract(solB1, 0); V_pre, _ = extract(solB1, -1)
V_post, _ = extract(solB2, 0); V_rec, m_rec = extract(solB2, -1)
rows.append(([V0_B, V_pre, V_post, V_rec], ["t=0", "t=200, before", "t=200, after", "t=700"], m_rec, "memory, t=700"))
# Row C - cancer-like autonomy
V0_C, _ = extract(solC1, 0); V_prec, _ = extract(solC1, -1)
V_canc0, _ = extract(solC2, 0); V_canc, m_canc = extract(solC2, -1)
rows.append(([V0_C, V_prec, V_canc0, V_canc], ["t=0", "t=200, before", "t=200, trigger", "t=700"], m_canc, "memory, t=700"))

for r, (Vs, ttl, mfin, mttl) in enumerate(rows):
    for c in range(4):
        imV = show(fig.add_subplot(gs[r, c]), Vs[c], cmap_V, vmin_V, vmax_V, ttl[c])
    imm = show(fig.add_subplot(gs[r, 6]), mfin, cmap_m, vmin_m, vmax_m, mttl)
    if r == 0:
        cbV = fig.colorbar(imV, cax=fig.add_subplot(gs[:, 4]))
        cbV.set_label(r"$V_i$", fontsize=7); cbV.ax.tick_params(labelsize=6)
        cbm = fig.colorbar(imm, cax=fig.add_subplot(gs[:, 7]))
        cbm.set_label(r"$m_i$", fontsize=7); cbm.ax.tick_params(labelsize=6)

for row, lbl in enumerate(["(A) Target\nconvergence", "(B) Pattern\nrecovery", "(C) Cancer-like\nautonomy"]):
    fig.text(0.01, 0.82 - row*0.31, lbl, fontsize=7.5, fontweight="bold", va="center", rotation=0)

import os
out = os.path.join(os.path.dirname(__file__), "figs", "fig_2d_simulation.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")

# ── Lyapunov trajectory plot ─────────────────────────────────────────────────
def H_val(y):
    """H_ID(V; G) for one snapshot: sum U + (alpha/2) sum (V - phi(IDENT))^2 + (1/2) V^T L_G V."""
    V = y[:NC]; G = y[NC:NC+NPAIRS]; m = y[NC+NPAIRS:]
    U = a * V**4 / 4 - a * L**2 * V**2 / 2          # U(V) s.t. U'=f
    phi_i = target_phi(IDENT)
    H = np.sum(U) + (alpha/2)*np.sum((V - phi_i)**2)
    # gap-junction (Laplacian) term
    H += 0.5 * np.sum(G * (V[PI] - V[PJ])**2)
    return H

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6})
fig2, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
fig2.patch.set_facecolor("white")

# A: H vs t
Hvals_A = np.array([H_val(solA.y[:, k]) for k in range(len(tA))])
axes[0].plot(tA, Hvals_A, color="steelblue", lw=1.5)
axes[0].set_xlabel("time"); axes[0].set_ylabel(r"$\mathcal{H}_{\rm ID}$")
axes[0].set_title(r"(A) Target convergence")
axes[0].axhline(Hvals_A[-1], color="grey", ls="--", lw=0.8, label="final value")
axes[0].legend()

# B: H across both phases
tB_all = np.concatenate([tB1, tB2])
Hvals_B = np.array(
    [H_val(solB1.y[:, k]) for k in range(len(tB1))] +
    [H_val(solB2.y[:, k]) for k in range(len(tB2))]
)
axes[1].plot(tB_all, Hvals_B, color="seagreen", lw=1.5)
axes[1].axvline(200, color="tomato", ls="--", lw=1, label="perturbation")
axes[1].set_xlabel("time"); axes[1].set_ylabel(r"$\mathcal{H}_{\rm ID}$")
axes[1].set_title(r"(B) Pattern recovery")
axes[1].legend()

# C: H across both phases
tC_all = np.concatenate([tC1, tC2])
Hvals_C = np.array(
    [H_val(solC1.y[:, k]) for k in range(len(tC1))] +
    [H_val(solC2.y[:, k]) for k in range(len(tC2))]
)
axes[2].plot(tC_all, Hvals_C, color="tomato", lw=1.5)
axes[2].axvline(200, color="navy", ls="--", lw=1, label="G loss")
axes[2].set_xlabel("time"); axes[2].set_ylabel(r"$\mathcal{H}_{\rm ID}$")
axes[2].set_title(r"(C) Cancer-like autonomy")
axes[2].legend()

fig2.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), "figs", "fig_2d_lyapunov.pdf")
fig2.savefig(out2, dpi=180, bbox_inches="tight")
print(f"Saved → {out2}")
print("Done.")
