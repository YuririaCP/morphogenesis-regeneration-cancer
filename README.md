# Code for "Morphogenesis, Regeneration, and Cancer-Like Autonomy in a Common Bioelectric Dynamical System"

Simulation code, result files, and figure scripts for the manuscript by Y. Cortés-Poza.

## Requirements

Python 3.9 or later with the packages in `requirements.txt` (tested with Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4). No other dependencies.

```bash
pip install -r requirements.txt
cd scripts
QUICK=1 python3 sim_regimes.py     # short smoke test of any simulation script
```

Every `sim_*.py` script accepts `QUICK=1` for a short run (its output goes to a `figs/` folder next to the script, so the figures in the top-level `figs/` are not overwritten). Full runs write figures to `../figs/`.

## Contents of `scripts/`

| File | Purpose | Output |
|---|---|---|
| `minimal_model.py` | Model engine (V, m, G, occupancy; RK4) | imported by all scripts |
| `sim_regen_spont.py`, `fig_regen_spont.py` | Spontaneous formation, regeneration sweep, ablations (main Fig. 3, Table S1) | `regen_spont_results.json` |
| `sim_regen_interface_example.py` | Scope control: one lesion across one polarity interface (seed 7) | printed |
| `sim_regimes.py`, `sim_autonomy.py` | Recovery maps; fixed conductances and adaptive isolation | figures |
| `sim_noise.py` | Noise experiments, escape times (Fig. S1) | figures, tables |
| `sim_seeds.py`, `sim_seeds_env.py` | Robustness across 16 seeds and environment control | `seeds_results.jsonl`, `seeds_env_results.json` |
| `sim_blockade.py`, `sim_blockade_ablation.py`, `fig_blockade.py` | Transient coupling blockade (Prediction 3, Fig. S9) | `blockade_results.jsonl`, `blockade_ablation_results.jsonl` |
| `sim2d*.py` | Two-dimensional prescribed-target extension | figures |
| `prove_threshold.py` | Symbolic and numerical check of Lemma 1 | printed |
| `make_supp_tables.py` | Tables of the Supplementary Material | LaTeX |

`figs/` holds the final figures used in the manuscript.

## Notes

- Pre-computed result files are included so figures and tables can be regenerated without rerunning the sweeps.
- Full runs of `sim_seeds_env.py` and the 2D scripts take several minutes to hours depending on the machine.
- A design of the blockade experiment that clamped the stored conductance was discarded (it erased the adaptive state); the final design multiplies the transmission through the affected edges by 0.03 while the stored conductance keeps adapting.

## License and citation

License: to be chosen by the author before public release. Citation: see the manuscript; a versioned release will be archived at Zenodo (DOI to be added).
