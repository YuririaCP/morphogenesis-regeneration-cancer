"""
minimal_model.py
Yuriria Cortés-Poza
====================
Minimal unified dynamical model for morphogenesis, regeneration, and cancer.

State variables per occupied cell i (ni = 1):
  V_i  : membrane potential : bistable cubic dynamics  (fast)
  m_i  : morphological memory variable                  (slow)

Edge variables:
  G_ij : gap-junction conductance, symmetric            (intermediate)

Discrete:
  n_i  : occupancy in {0, 1}

Governing equations
-------------------
Bioelectric layer (fast, timescale tau_V):
  C dV_i/dt = -f(V_i)
              + sum_{j~i} G_ij (V_j - V_i)
              - alpha * (V_i - phi(m_i))

Memory layer (slow, timescale tau_m >> tau_V):
  tau_m dm_i/dt = lam_m * (sigma_h(V_i) - m_i)
                  + kappa * sum_{j~i} G_ij (m_j - m_i)
  (normalised production: [0,1]^N is positively invariant)

Coupling dynamics (intermediate, tau_V < tau_G < tau_m):
  tau_G dG_ij/dt = sigma_G(V_i)*sigma_G(V_j)*(1 + gamma*m_i*m_j) - lam_G*G_ij

Occupancy rule (discrete, checked every dt_n):
  n_i becomes 1 iff n_i=0 AND #{occupied neighbors} >= q
                            AND mean(V_j, j occupied neighbor) > V_0
  New cell inherits: V_i <- mean(V_j), m_i <- mean(m_j)

Three regimes (same equations, different initial conditions / parameter values):
  Morphogenesis : ni=1 everywhere, V disordered, m ~ 0, G uniform
  Regeneration  : established pattern, then ni=0 in lesion D
  Cancer        : established pattern, then G_ij reduced in region Omega
"""

import os
import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =============================================================================
#  PARAMETERS
# =============================================================================

@dataclass
class MinimalParams:
    """
    Parameters for the minimal 4-variable model.
    All timescales normalized to tau_V = 1.
    """
    # --- Geometry ---
    N:   int   = 60     # number of cells (1D chain)
    dim: int   = 1

    # --- Bistable bioelectric layer ---
    C:       float = 1.0
    a_bist:  float = 1.0
    V_minus: float = -1.5
    V_0:     float =  0.0
    V_plus:  float =  1.5

    # --- Error-correction coupling to memory ---
    alpha: float = 0.30    # strength of Vi -> phi(mi) pull

    # --- Memory layer ---
    tau_m:  float = 80.0   # slow timescale
    lam_m:  float = 0.8    # memory relaxation rate (T_m = tau_m/lam_m = 100)
    kappa:  float = 0.10   # spatial memory diffusion coefficient
    k_phi:  float = 4.0    # steepness of phi(m): maps m -> voltage target
    k_h:    float = 4.0    # steepness of sigma_h(V): maps V -> memory drive
    k_G:    float = 5.0    # steepness of sigma_G(V): maps V -> conductance production

    # --- Gap-junction coupling ---
    G_init:  float = 0.50   # initial conductance (uniform)
    tau_G:   float = 5.0
    lam_G:   float = 0.10
    gamma_m: float = 0.60   # memory-coherence contribution to G growth

    # --- Occupancy (growth / repair) ---
    q:           int   = 1      # min occupied neighbors required for birth
    dt_n:        float = 1.0    # occupancy update interval

    # --- Numerical ---
    dt:   float = 0.02   # RK4 time step
    seed: int   = 42


# =============================================================================
#  ACTIVATION FUNCTIONS
# =============================================================================

def f_bist(V, a, Vm, V0, Vp):
    """Bistable cubic current: f(V) = a(V-V-)(V-V0)(V-V+)."""
    return a * (V - Vm) * (V - V0) * (V - Vp)

def sigma(x, x0=0.0, k=5.0):
    """Logistic sigmoid, numerically stable."""
    z = np.clip(-k * (x - x0), -500, 500)
    return 1.0 / (1.0 + np.exp(z))

def phi_func(m, V_minus, V_plus, k_phi):
    """
    phi(m): memory -> voltage target.
    phi in [V_minus, V_plus], monotone increasing in m.
    """
    s = sigma(m, x0=0.5, k=k_phi)
    return V_minus + (V_plus - V_minus) * s

def sigma_h(V, V_0, k_h):
    """sigma_h(V): voltage -> memory drive, in [0,1]."""
    return sigma(V, x0=V_0, k=k_h)


# =============================================================================
#  NETWORK CLASS
# =============================================================================

class MinimalNetwork:
    """
    Minimal bioelectric network: state per cell = (V_i, m_i), plus G_ij, n_i.

    The three regimes: morphogenesis, regeneration, cancer: are obtained
    by different initialization protocols; the equations are unchanged.
    """

    def __init__(self, p: MinimalParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self._build_geometry()
        self._initialize_state()
        self.time = 0.0
        # history for plotting
        self.hist_V  = []   # list of V snapshots
        self.hist_m  = []   # list of m snapshots
        self.hist_G  = []   # list of mean G per cell
        self.hist_n  = []
        self.hist_t  = []
        self._next_n_update = p.dt_n

    # ------------------------------------------------------------------
    #  GEOMETRY
    # ------------------------------------------------------------------

    def _build_geometry(self):
        p = self.p
        self.N_cells = p.N
        # neighbors list
        self.neighbors = [[] for _ in range(self.N_cells)]
        for i in range(self.N_cells):
            if i > 0:
                self.neighbors[i].append(i - 1)
            if i < self.N_cells - 1:
                self.neighbors[i].append(i + 1)
        # sparse edge list
        self.edges = []
        for i in range(self.N_cells - 1):
            self.edges.append((i, i + 1))

    def _edge_index(self):
        """Return arrays row_i, row_j for all edges (both directions)."""
        ri, rj = [], []
        for (i, j) in self.edges:
            ri += [i, j]; rj += [j, i]
        return np.array(ri), np.array(rj)

    # ------------------------------------------------------------------
    #  INITIALIZATION
    # ------------------------------------------------------------------

    def _initialize_state(self):
        p = self.p
        N = self.N_cells
        # Continuous state
        self.V = self.rng.uniform(p.V_minus, p.V_plus, N)
        self.m = self.rng.uniform(0.0, 0.1, N)   # nearly uncommitted
        # Gap-junction conductance matrix (full N×N, symmetric)
        self.G = np.zeros((N, N))
        for (i, j) in self.edges:
            self.G[i, j] = self.G[j, i] = p.G_init
        # Occupancy
        self.n = np.ones(N, dtype=int)

    # ------------------------------------------------------------------
    #  DERIVATIVES
    # ------------------------------------------------------------------

    def _derivatives(self, V, m, G, n):
        """Vectorized derivatives for all three layers."""
        p = self.p
        N = self.N_cells

        # Clip V to physical range for numerical stability
        V = np.clip(V, p.V_minus - 0.5, p.V_plus + 0.5)

        # Weighted graph Laplacian action via matrix multiply (vectorized)
        n_mask = n.astype(float)  # 1.0 for occupied, 0.0 for empty

        # Gap current for V: sum_j G_ij (V_j - V_i) = (G @ V) - deg_G * V
        # where deg_G[i] = sum_j G_ij
        GV    = G @ (V * n_mask) - (G @ n_mask) * V   # shape (N,)
        Gm    = G @ (m * n_mask) - (G @ n_mask) * m   # shape (N,)

        # Bistable current
        f_V = f_bist(V, p.a_bist, p.V_minus, p.V_0, p.V_plus)

        # Sigmoidal activations
        sg  = sigma(V, x0=p.V_0, k=p.k_G)
        sh  = sigma_h(V, p.V_0, p.k_h)
        phi = phi_func(m, p.V_minus, p.V_plus, p.k_phi)

        # Bioelectric dynamics
        dV = n_mask * (-f_V + GV - p.alpha * (V - phi)) / p.C

        # Memory dynamics (normalised production: keeps [0,1]^N positively invariant)
        dm = n_mask * (p.lam_m * (sh - m) + p.kappa * Gm) / p.tau_m

        # Coupling dynamics (vectorized outer product)
        # drive_ij = sg_i * sg_j * (1 + gamma * m_i * m_j)
        sg_mat = np.outer(sg, sg)           # (N, N)
        m_mat  = np.outer(m,  m)            # (N, N)
        drive  = sg_mat * (1.0 + p.gamma_m * m_mat)
        # Only update existing edges between occupied cells
        edge_mask = (G > 0).astype(float)   # existing edges
        occ_pair  = np.outer(n_mask, n_mask)
        dG = edge_mask * occ_pair * (drive - p.lam_G * G) / p.tau_G

        return dV, dm, dG

    # ------------------------------------------------------------------
    #  RK4 STEP
    # ------------------------------------------------------------------

    def _rk4_step(self):
        p = self.p
        V, m, G, n = self.V, self.m, self.G, self.n
        dt = p.dt

        k1V, k1m, k1G = self._derivatives(V, m, G, n)

        V2 = V + 0.5*dt*k1V; m2 = m + 0.5*dt*k1m; G2 = np.clip(G + 0.5*dt*k1G, 0, None)
        k2V, k2m, k2G = self._derivatives(V2, m2, G2, n)

        V3 = V + 0.5*dt*k2V; m3 = m + 0.5*dt*k2m; G3 = np.clip(G + 0.5*dt*k2G, 0, None)
        k3V, k3m, k3G = self._derivatives(V3, m3, G3, n)

        V4 = V + dt*k3V; m4 = m + dt*k3m; G4 = np.clip(G + dt*k3G, 0, None)
        k4V, k4m, k4G = self._derivatives(V4, m4, G4, n)

        p = self.p
        self.V = np.clip(V + (dt/6.0)*(k1V + 2*k2V + 2*k3V + k4V),
                         p.V_minus - 0.5, p.V_plus + 0.5)
        self.m = np.clip(m + (dt/6.0)*(k1m + 2*k2m + 2*k3m + k4m), 0.0, 1.0)
        self.G = np.clip(G + (dt/6.0)*(k1G + 2*k2G + 2*k3G + k4G), 0.0, 2.0)

    # ------------------------------------------------------------------
    #  OCCUPANCY UPDATE
    # ------------------------------------------------------------------

    def _update_occupancy(self):
        p = self.p
        N = self.N_cells
        new_n = self.n.copy()
        new_V = self.V.copy()
        new_m = self.m.copy()

        for i in range(N):
            if self.n[i] == 0:
                occ_nbrs = [j for j in self.neighbors[i] if self.n[j] == 1]
                if len(occ_nbrs) >= p.q:
                    V_nbr = np.mean(self.V[occ_nbrs])
                    if V_nbr > p.V_0:  # depolarized neighbors drive growth
                        new_n[i] = 1
                        new_V[i] = V_nbr
                        new_m[i] = np.mean(self.m[occ_nbrs])
                        # initialize G to neighbors
                        for j in occ_nbrs:
                            self.G[i, j] = self.G[j, i] = p.G_init * 0.5

        self.n  = new_n
        self.V  = new_V
        self.m  = new_m

    # ------------------------------------------------------------------
    #  RUN
    # ------------------------------------------------------------------

    def run(self, T_total: float, record_every: float = 1.0):
        """Integrate the system for T_total time units."""
        t = self.time
        t_end = t + T_total
        next_record = t

        while t < t_end:
            # occupancy update
            if t >= self._next_n_update:
                self._update_occupancy()
                self._next_n_update += self.p.dt_n

            # RK4 step
            self._rk4_step()
            t += self.p.dt

            # record
            if t >= next_record:
                self.hist_t.append(t)
                self.hist_V.append(self.V.copy())
                self.hist_m.append(self.m.copy())
                # mean conductance per cell
                G_mean = np.array([
                    np.mean([self.G[i, j] for j in self.neighbors[i]])
                    if self.neighbors[i] else 0.0
                    for i in range(self.N_cells)
                ])
                self.hist_G.append(G_mean)
                self.hist_n.append(self.n.copy())
                next_record += record_every

        self.time = t

    # ------------------------------------------------------------------
    #  HELPER METHODS
    # ------------------------------------------------------------------

    def impose_lesion(self, i_start: int, i_end: int):
        """Set ni=0 (and V, m to zero) in cells [i_start, i_end)."""
        self.n[i_start:i_end] = 0
        self.V[i_start:i_end] = 0.0
        self.m[i_start:i_end] = 0.0
        for i in range(i_start, i_end):
            for j in self.neighbors[i]:
                self.G[i, j] = self.G[j, i] = 0.0

    def reduce_coupling(self, i_start: int, i_end: int, G_cancer: float = 0.02):
        """
        Reduce G_ij in a region [i_start, i_end) to G_cancer.
        This models loss of gap-junction communication (cancer regime).
        """
        for i in range(i_start, i_end):
            for j in self.neighbors[i]:
                self.G[i, j] = self.G[j, i] = G_cancer
            # also reduce edges within the region to neighbors outside
        # Specifically cut boundary edges
        if i_start > 0:
            self.G[i_start, i_start-1] = self.G[i_start-1, i_start] = G_cancer
        if i_end < self.N_cells:
            self.G[i_end-1, i_end] = self.G[i_end, i_end-1] = G_cancer

    def coherence_score(self) -> float:
        """
        Global coherence: fraction of active cells in the majority voltage state.
        1.0 = fully patterned (all cells either V+ or V-, no intermediate).
        """
        active = self.n == 1
        if active.sum() == 0:
            return 0.0
        V_active = self.V[active]
        n_dep = np.sum(V_active > self.p.V_0)
        n_hyp = np.sum(V_active <= self.p.V_0)
        # coherence: how bimodal is the distribution?
        frac = max(n_dep, n_hyp) / active.sum()
        return float(frac)

    def fragmentation_index(self, omega_start: int, omega_end: int) -> float:
        """
        Measure how much the voltage in region Omega diverges from the
        global pattern. Returns 0 if Omega matches global pattern, 1 if inverted.
        """
        active = self.n == 1
        if active.sum() == 0:
            return 0.0
        V_global = np.mean(self.V[active])
        omega_mask = np.zeros(self.N_cells, dtype=bool)
        omega_mask[omega_start:omega_end] = True
        omega_active = omega_mask & active
        if omega_active.sum() == 0:
            return 0.0
        V_omega = np.mean(self.V[omega_active])
        # normalized difference
        scale = self.p.V_plus - self.p.V_minus
        return float(abs(V_omega - V_global) / scale)


# =============================================================================
#  EXPERIMENT PROTOCOLS
# =============================================================================

def run_morphogenesis(p: MinimalParams, T: float = 150.0):
    """
    Morphogenesis protocol:
    - All cells occupied, V disordered, m nearly uncommitted.
    - System converges to a coherent global pattern.
    """
    net = MinimalNetwork(p)
    net.run(T, record_every=1.0)
    return net


def _find_depolarised_omega(V, N, V0, omega_size):
    """
    Find a block of `omega_size` consecutive depolarised cells (V_i > V0),
    avoiding the first and last 2 cells (boundary effects).
    Returns (i0, i1) or falls back to a central guess.
    """
    # Collect depolarised indices
    dep = [i for i in range(2, N - 2) if V[i] > V0]
    if len(dep) < omega_size:
        # Fall back: use the region with highest mean V
        best_i0 = 2
        best_mean = -1e9
        for start in range(2, N - omega_size - 2):
            m = np.mean(V[start:start + omega_size])
            if m > best_mean:
                best_mean = m
                best_i0 = start
        return best_i0, best_i0 + omega_size

    # Find the largest contiguous run of depolarised cells
    best_start, best_len, cur_start, cur_len = dep[0], 1, dep[0], 1
    for k in range(1, len(dep)):
        if dep[k] == dep[k - 1] + 1:
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start = dep[k]
            cur_len = 1

    if best_len < omega_size:
        # use any depolarised cells
        i0 = dep[0]
        return i0, min(i0 + omega_size, N - 2)

    # Place Omega in the middle of the largest depolarised block
    center = best_start + best_len // 2
    i0 = max(best_start + 1, center - omega_size // 2)
    i1 = min(best_start + best_len - 1, i0 + omega_size)
    return int(i0), int(i1)


def run_regeneration(p: MinimalParams, T_pre: float = 120.0,
                     T_post: float = 120.0,
                     lesion_frac: float = 0.25):
    """
    Regeneration protocol:
    - Establish a morphogenetic attractor using a structured left/right
      initial condition (left half depolarised, right half hyperpolarised).
      This guarantees a wide depolarised zone regardless of random seed,
      and ensures boundary cells are at V_+ so the occupancy rule fires.
    - Impose lesion at fixed cells in the depolarised zone.
    - Run to recovery (T_post).
    Returns: net, i0, i1, t_lesion_idx.
    """
    import copy
    N = p.N
    half = N // 2

    # Phase 1: morphogenesis with structured initial condition.
    # Left half depolarised, right half hyperpolarised: guarantees a wide
    # depolarised zone regardless of random seed so the lesion boundary cells
    # are always at V_+ and the occupancy rule fires.
    net = MinimalNetwork(p)
    net.V[:half] = p.V_plus  * 0.97
    net.V[half:] = p.V_minus * 0.97
    net.m[:] = np.ones(N) * 0.05
    for (ci, cj) in net.edges:
        net.G[ci, cj] = net.G[cj, ci] = p.G_init
    net.run(T_pre, record_every=1.0)

    # Use slower occupancy update (dt_n = 5) in the recovery phase so that
    # the repopulation wave is visible on the time-axis (otherwise cells fill
    # in within ~7 time units, <3% of the plot, invisible).
    # This is biologically realistic: cell migration/division takes several
    # membrane-relaxation times.
    p_recovery = copy.copy(p)
    p_recovery.dt_n = 5.0

    # Re-create network keeping the established state, with slower dt_n.
    net2 = MinimalNetwork(p_recovery)
    net2.V = net.V.copy()
    net2.m = net.m.copy()
    net2.G = net.G.copy()
    net2.n = net.n.copy()
    net2.time = net.time
    net2._next_n_update = net.time + p_recovery.dt_n
    net2.hist_V = [s.copy() for s in net.hist_V]
    net2.hist_m = [s.copy() for s in net.hist_m]
    net2.hist_G = [s.copy() for s in net.hist_G]
    net2.hist_n = [s.copy() for s in net.hist_n]
    net2.hist_t = list(net.hist_t)

    # Lesion: 15-cell block centred in depolarised half
    omega_size = int(N * lesion_frac)
    i0 = half // 2 - omega_size // 2
    i0 = max(2, i0)
    i1 = i0 + omega_size
    t_lesion = len(net2.hist_t)
    net2.impose_lesion(i0, i1)

    # Phase 2: regeneration (slow repopulation, visible recovery wave)
    net2.run(T_post, record_every=1.0)
    return net2, i0, i1, t_lesion


def run_cancer(p: MinimalParams, T_pre: float = 120.0,
               T_post: float = 120.0,
               cancer_frac: float = 0.25,
               G_cancer: float = 0.015):
    """
    Cancer protocol:
    - Run to established pattern (T_pre).
    - Reduce coupling in Omega (G -> G_cancer < G_c).
    - Invert V in Omega (V_i -> -V_i): simulates a voltage perturbation
      (de-differentiation event) in a decoupled region.
    - With beta_eff = G_cancer + alpha < alpha_c*, the tissue CANNOT restore
      the correct voltage; Omega settles to the cancer-regime attractor
      V_c ~ 1.30 (Lemma lem:mono), distinct from the global pattern.
    Returns: net, omega boundaries, index of perturbation.
    """
    net = MinimalNetwork(p)
    net.run(T_pre, record_every=1.0)

    N = p.N
    i0 = int(N * 0.35)
    i1 = int(N * 0.35 + N * cancer_frac)
    t_cancer = len(net.hist_t)

    # Step 1: isolate Omega (reduce coupling on all incident edges)
    net.reduce_coupling(i0, i1, G_cancer=G_cancer)

    # Step 2: invert voltage in Omega: local memory (m_i) stays intact,
    # so phi(m_i) still points to the original polarity.  With
    # beta_eff = G_cancer + alpha < alpha_c*, the bistable landscape
    # has a coexisting local attractor near the inverted state; the global
    # field is too weak to correct it.
    net.V[i0:i1] = -net.V[i0:i1]
    net.V = np.clip(net.V, p.V_minus, p.V_plus)

    net.run(T_post, record_every=1.0)
    return net, i0, i1, t_cancer


# =============================================================================
#  PHASE DIAGRAM
# =============================================================================

def compute_phase_diagram(G_vals, alpha_vals, T: float = 120.0,
                          N: int = 40, seed: int = 42):
    """
    Sweep (G_init, alpha) and compute coherence score at final time.
    Returns 2D array of coherence scores.
    """
    scores = np.zeros((len(alpha_vals), len(G_vals)))
    for ai, alpha in enumerate(alpha_vals):
        for gi, G_init in enumerate(G_vals):
            p = MinimalParams(N=N, G_init=G_init, alpha=alpha, seed=seed)
            net = MinimalNetwork(p)
            net.run(T, record_every=T)  # only record final state
            scores[ai, gi] = net.coherence_score()
    return scores


# =============================================================================
#  FIGURE GENERATION
# =============================================================================

# Custom colormaps
_vmap = LinearSegmentedColormap.from_list(
    'bioelectric', ['#2166ac', '#f7f7f7', '#d6604d'], N=256)
_mmap = LinearSegmentedColormap.from_list(
    'memory', ['#f0f0f0', '#253494'], N=256)


def _space_time_panel(ax, hist, t_arr, vmin, vmax, cmap, ylabel, xlabel=True):
    """Plot a space-time heatmap from a list of snapshots."""
    M = np.array(hist).T   # shape (N_cells, T)
    im = ax.imshow(M, aspect='auto', origin='lower',
                   extent=[t_arr[0], t_arr[-1], 0, M.shape[0]],
                   vmin=vmin, vmax=vmax, cmap=cmap, interpolation='bilinear')
    ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel('Time', fontsize=9)
    return im


def fig_morphogenesis(net: MinimalNetwork, savepath: str):
    """Figure 2: Morphogenesis space-time plots."""
    t_arr = np.array(net.hist_t)
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.0), sharex=True,
                             gridspec_kw={'hspace': 0.38})

    # V(x, t)
    im0 = _space_time_panel(axes[0], net.hist_V, t_arr,
                             net.p.V_minus, net.p.V_plus, _vmap,
                             r'Cell index $i$', xlabel=False)
    axes[0].set_title(r'Membrane potential $V_i(t)$', fontsize=9, pad=3)
    plt.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.03,
                 label=r'$V_i$')

    # m(x, t)
    im1 = _space_time_panel(axes[1], net.hist_m, t_arr,
                             0.0, 1.0, _mmap,
                             r'Cell index $i$', xlabel=False)
    axes[1].set_title(r'Morphological memory $m_i(t)$', fontsize=9, pad=3)
    plt.colorbar(im1, ax=axes[1], pad=0.01, fraction=0.03, label=r'$m_i$')

    # mean G(x, t)
    im2 = _space_time_panel(axes[2], net.hist_G, t_arr,
                             0.0, 1.2, 'YlOrBr',
                             r'Cell index $i$', xlabel=True)
    axes[2].set_title(r'Mean gap-junction conductance $\bar{G}_i(t)$',
                      fontsize=9, pad=3)
    plt.colorbar(im2, ax=axes[2], pad=0.01, fraction=0.03, label=r'$\bar{G}_{ij}$')
    axes[2].set_xlabel('Time', fontsize=9)

    fig.suptitle('Spontaneous formation and pinning of bioelectric domains',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


def fig_regeneration(net: MinimalNetwork, i0: int, i1: int,
                     t_lesion_idx: int, savepath: str):
    """Figure 3: Regeneration space-time plots with lesion marker."""
    t_arr = np.array(net.hist_t)
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                             gridspec_kw={'hspace': 0.38})

    im0 = _space_time_panel(axes[0], net.hist_V, t_arr,
                             net.p.V_minus, net.p.V_plus, _vmap,
                             r'Cell index $i$', xlabel=False)
    axes[0].set_title(r'$V_i(t)$: Regeneration after lesion', fontsize=9, pad=4)
    plt.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.03, label=r'$V_i$')

    im1 = _space_time_panel(axes[1], net.hist_m, t_arr,
                             0.0, 1.0, _mmap,
                             r'Cell index $i$', xlabel=True)
    axes[1].set_title(r'$m_i(t)$', fontsize=9, pad=4)
    plt.colorbar(im1, ax=axes[1], pad=0.01, fraction=0.03, label=r'$m_i$')
    axes[1].set_xlabel('Time', fontsize=9)

    # Mark lesion region and time
    t_les = t_arr[t_lesion_idx] if t_lesion_idx < len(t_arr) else t_arr[-1]
    for ax in axes:
        ax.axvline(t_les, color='red', lw=1.5, ls='--', alpha=0.8, label='lesion')
        ax.axhspan(i0, i1, color='red', alpha=0.08)

    axes[0].legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


def fig_cancer(net: MinimalNetwork, i0: int, i1: int,
               t_cancer_idx: int, savepath: str):
    """Figure 4: Cancer as attractor fragmentation."""
    t_arr = np.array(net.hist_t)
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                             gridspec_kw={'hspace': 0.38})

    im0 = _space_time_panel(axes[0], net.hist_V, t_arr,
                             net.p.V_minus, net.p.V_plus, _vmap,
                             r'Cell index $i$', xlabel=False)
    axes[0].set_title(r'$V_i(t)$: Cancer-like autonomy, local attractor in $\Omega$',
                      fontsize=9, pad=4)
    plt.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.03, label=r'$V_i$')

    im1 = _space_time_panel(axes[1], net.hist_G, t_arr,
                             0.0, 1.2, 'YlOrBr',
                             r'Cell index $i$', xlabel=True)
    axes[1].set_title(r'$\bar{G}_i(t)$: conductance loss in $\Omega$',
                      fontsize=9, pad=4)
    plt.colorbar(im1, ax=axes[1], pad=0.01, fraction=0.03, label=r'$\bar{G}_{ij}$')
    axes[1].set_xlabel('Time', fontsize=9)

    t_can = t_arr[t_cancer_idx] if t_cancer_idx < len(t_arr) else t_arr[-1]
    for ax in axes:
        ax.axvline(t_can, color='darkorange', lw=1.5, ls='--', alpha=0.9,
                   label=r'coupling loss in $\Omega$')
        ax.axhspan(i0, i1, color='darkorange', alpha=0.10)

    # Place legend in both panels so both show the coupling-loss marker
    for ax in axes:
        ax.legend(fontsize=8, loc='upper left',
                  framealpha=0.85, edgecolor='0.7')
    plt.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


def fig_phase_diagram(scores, G_vals, alpha_vals, savepath: str):
    """Figure 5: Phase diagram in (G_init, alpha) space."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(scores, origin='lower', aspect='auto',
                   extent=[G_vals[0], G_vals[-1], alpha_vals[0], alpha_vals[-1]],
                   vmin=0.5, vmax=1.0, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Majority-phase fraction')
    ax.set_xlabel(r'Mean coupling $\bar{G}$', fontsize=10)
    ax.set_ylabel(r'Error-correction strength $\alpha$', fontsize=10)
    ax.set_title('Phase dominance in coupling-error-correction space', fontsize=9)

    # Draw approximate phase boundary (coherence = 0.75)
    from matplotlib.contour import QuadContourSet
    G_grid, a_grid = np.meshgrid(G_vals, alpha_vals)
    try:
        cs = ax.contour(G_grid, a_grid, scores, levels=[0.75],
                        colors=['black'], linewidths=1.5, linestyles='--')
        ax.clabel(cs, fmt='%.2f', fontsize=8)
    except Exception:
        pass

    # Annotate regions
    ax.text(0.19, 0.05, 'Multidomain', fontsize=7.5,
            color='white', ha='center', va='bottom',
            bbox=dict(boxstyle='square,pad=0.1', facecolor='none',
                      edgecolor='none'))
    ax.text(0.68, 0.40, 'Dominant\nglobal phase', fontsize=7.5,
            color='white', ha='center', va='bottom',
            fontweight='bold')

    plt.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


def fig_bistability(p: MinimalParams, savepath: str):
    """Figure 1B: Double-well potential and bistable phase portrait."""
    V_vals = np.linspace(p.V_minus - 0.3, p.V_plus + 0.3, 500)
    f_vals = f_bist(V_vals, p.a_bist, p.V_minus, p.V_0, p.V_plus)
    # Potential U(V) = ∫f dV  (minima at stable states V_±, maximum at V_0)
    dV = V_vals[1] - V_vals[0]
    U_vals = np.cumsum(f_vals) * dV
    U_vals -= U_vals.min()

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Phase portrait
    axes[0].plot(V_vals, f_vals, 'k-', lw=1.8)
    axes[0].axhline(0, color='gray', lw=0.8, ls='--')
    axes[0].axvline(p.V_minus, color='#2166ac', lw=1.2, ls=':')
    axes[0].axvline(p.V_plus,  color='#d6604d', lw=1.2, ls=':')
    axes[0].axvline(p.V_0,     color='gray',    lw=1.0, ls=':')
    axes[0].scatter([p.V_minus, p.V_plus], [0, 0],
                    color=['#2166ac', '#d6604d'], zorder=5, s=50)
    axes[0].scatter([p.V_0], [0], color='gray', marker='o',
                    facecolors='none', zorder=5, s=50)
    axes[0].set_xlabel(r'$V$', fontsize=10)
    axes[0].set_ylabel(r'$-f(V)$', fontsize=10)
    axes[0].set_title('Phase portrait', fontsize=9)
    axes[0].text(p.V_minus - 0.05, 0.15, r'$V_-$', color='#2166ac',
                 fontsize=9, ha='right')
    axes[0].text(p.V_plus + 0.05, 0.15, r'$V_+$', color='#d6604d', fontsize=9)
    axes[0].text(p.V_0, 0.18, r'$V_0$', color='gray', fontsize=9, ha='center')

    # Potential  (U = ∫f dV  has minima at V_±  → genuine double-well)
    axes[1].plot(V_vals, U_vals, 'k-', lw=1.8)
    idx_Vm = np.argmin(np.abs(V_vals - p.V_minus))
    idx_Vp = np.argmin(np.abs(V_vals - p.V_plus))
    idx_V0 = np.argmin(np.abs(V_vals - p.V_0))
    axes[1].scatter([p.V_minus, p.V_plus],
                    [U_vals[idx_Vm], U_vals[idx_Vp]],
                    color=['#2166ac', '#d6604d'], zorder=5, s=50)
    axes[1].scatter([p.V_0], [U_vals[idx_V0]],
                    color='gray', marker='o', facecolors='none', zorder=5, s=50)
    axes[1].set_xlabel(r'$V$', fontsize=10)
    axes[1].set_ylabel(r'$U(V)$', fontsize=10)
    axes[1].set_title('Double-well potential', fontsize=9)
    label_y = U_vals.max() * 0.18
    axes[1].text(p.V_minus, label_y, r'$V_-$', color='#2166ac',
                 fontsize=9, ha='center')
    axes[1].text(p.V_plus,  label_y, r'$V_+$', color='#d6604d',
                 fontsize=9, ha='center')
    axes[1].text(p.V_0, U_vals[idx_V0] + U_vals.max()*0.05,
                 r'$V_0$', color='gray', fontsize=9, ha='center')

    plt.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    os.makedirs('figs', exist_ok=True)
    p = MinimalParams(N=60, seed=7)

    print("=== Fig 1b: Bistability ===")
    fig_bistability(p, 'figs/fig_bistability.pdf')

    print("=== Fig 2: Morphogenesis ===")
    net_morph = run_morphogenesis(p, T=150.0)
    fig_morphogenesis(net_morph, 'figs/fig_morphogenesis.pdf')

    print("=== Fig 3: Regeneration ===")
    net_regen, i0r, i1r, t_les = run_regeneration(
        p, T_pre=120.0, T_post=120.0, lesion_frac=0.25)
    fig_regeneration(net_regen, i0r, i1r, t_les, 'figs/fig_regeneration.pdf')

    print("=== Fig 4: Cancer ===")
    net_can, i0c, i1c, t_can = run_cancer(
        p, T_pre=120.0, T_post=120.0, cancer_frac=0.25, G_cancer=0.015)
    fig_cancer(net_can, i0c, i1c, t_can, 'figs/fig_cancer.pdf')

    print("=== Fig 5: Phase diagram ===")
    G_vals     = np.linspace(0.05, 0.80, 10)
    alpha_vals = np.linspace(0.05, 0.60, 8)
    scores = compute_phase_diagram(G_vals, alpha_vals, T=80.0, N=30, seed=7)
    fig_phase_diagram(scores, G_vals, alpha_vals, 'figs/fig_phasediagram.pdf')

    print("\nAll figures saved to figs/")
