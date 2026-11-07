"""
fig_blockade.py
===============
Supplementary figure for Prediction 3: outcome of a transient transmission blockade of a 3-cell block as a
function of the blockade duration D (results of sim_blockade.py).
  (A) fraction of patterns in which the inverted block returns to the original sign, for the two orientations
      (depolarized block: the patterns that recover without blockade, D = 0).
  (B) mean target phi(m) over the block at release, versus D (mean and range over patterns), with the
      effective relaxation times T_G and T_m marked.
"""
import os, json, collections
import numpy as np
import matplotlib as _mpl; _mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # embeddable TrueType fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6})
R = [json.loads(l) for l in open(os.path.join(HERE, 'blockade_results.jsonl'))]
R = [r for r in R if not r.get('skipped')]
# duration analysis: depolarized-block patterns that already recover without blockade (D = 0)
BASE = {r['seed'] for r in R if r['orient'] == 'V+ to V-' and r['D'] == 0 and r['outcome'] == 'recovered'}
R = [r for r in R if r['orient'] != 'V+ to V-' or r['seed'] in BASE]
Ds = sorted({r['D'] for r in R})
T_G, T_m = 50.0, 100.0
cols = {'V+ to V-': '#1f5fa8', 'V- to V+': '#c0392b'}
labels = {'V+ to V-': 'depolarized block driven hyperpolarized', 'V- to V+': 'hyperpolarized block driven depolarized'}

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
a = ax[0]
for o in ('V+ to V-', 'V- to V+'):
    frac = []
    for D in Ds:
        rr = [r for r in R if r['orient'] == o and r['D'] == D]
        frac.append(np.mean([r['outcome'] == 'recovered' for r in rr]) if rr else np.nan)
    a.plot(Ds, frac, 'o-', ms=3, lw=1.1, color=cols[o], label=labels[o])
for x, t in ((T_G, '$T_G$'), (T_m, '$T_m$')):
    a.axvline(x, color='0.6', ls=':', lw=0.8); a.text(x + 2, 0.5, t, color='0.4', fontsize=6)
a.set_xlabel('blockade duration $D$'); a.set_ylabel('fraction of patterns recovered')
a.set_ylim(-0.05, 1.05); a.legend(loc='upper right', frameon=False); a.set_title('(A) outcome after release')

b = ax[1]
for o in ('V+ to V-', 'V- to V+'):
    mean, lo, hi = [], [], []
    for D in Ds:
        v = [r['phi_release'] for r in R if r['orient'] == o and r['D'] == D]
        mean.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
    b.plot(Ds, mean, 'o-', ms=3, lw=1.1, color=cols[o])
    b.fill_between(Ds, lo, hi, color=cols[o], alpha=0.15, lw=0)
b.axhline(0, color='0.5', lw=0.6)
for x in (T_G, T_m): b.axvline(x, color='0.6', ls=':', lw=0.8)
b.set_xlabel('blockade duration $D$'); b.set_ylabel(r'$\overline{\varphi(m)}$ over the block at release')
b.set_title('(B) memory target at release')
out = os.path.join(HERE, '..', 'figs', 'fig_blockade.pdf')
fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out)
