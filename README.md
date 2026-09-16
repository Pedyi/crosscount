# crosscount

Companion code for ***Predictions Buy Constants, Not Exponents: Streaming Higher-Order
Triadic Closure and the Ceiling on Learned Oracles.*** Submitted to *Data Mining and
Knowledge Discovery*.

The paper asks two questions about learning-augmented streaming subgraph counting. Can a
predictor ever change the space *exponent* on real data? And if not, what does it change?
The answers are: no, for a structural reason that is provable and cheap to check; and a
constant factor whose size is itself predictable.

This repository contains the estimator, the exact measurement machinery, and every
experiment in the paper.

## The primitive

A **cross-hyperedge triangle** is a triple `{u, v, w}` that is a triangle in the projection
of a hypergraph but whose three vertices never co-occur in a single hyperedge. It is the
higher-order triadic closure the simplicial-closure literature studies: the triangles a
large hyperedge drops on the projection are artefacts of projection, and this primitive
counts exactly the ones that are not.

## Install

```bash
git clone https://github.com/Pedyi/crosscount
cd crosscount
pip install -r requirements.txt
pip install -e .            # optional
```

Python 3.9+. No GPU, no compilation; the experiments run on a laptop or a free Colab CPU
runtime.

## Quick start

```python
from crosscount import CrossHG, evaluate, diagnostic_crossK3, cliques_plus_cross

# a synthetic hypergraph in the separation regime: copy-free cliques + shallow gadgets
hg = CrossHG(cliques_plus_cross(n_cliques=6, clique_size=40, n_gadgets=240))

print(diagnostic_crossK3(hg))   # kappa, kappa_copy, exact alpha_H, the two ratios
print(evaluate(hg, n_perm=50))  # localization, predictor, permutation null, shares
```

On your own data, pass any list of hyperedges (each a tuple of vertex ids):

```python
hg = CrossHG([(1, 2, 3), (2, 3, 4, 5), (1, 5), ...])
```

## Reproducing the paper

```bash
cd experiments
python run_all.py            # everything, in the order the paper reports it
```

or one table at a time:

| script | produces | paper |
|---|---|---|
| `run_diagnostic.py` | `table1_diagnostic.csv` | Table 1 — the diagnostic, exact on ten graphs |
| `run_main.py` | `table2_main.csv` | Table 2 — localization, predictor, permutation null |
| `run_learned.py` | `table3_learned_*.csv` | Table 3 — ridge model, leave-one-graph-out |
| `run_crossc4.py` | `table4_crossc4.csv` | Table 4 — cross-hyperedge four-cycles |
| `run_robustness.py` | `table5_*.csv` | robustness appendix |
| `run_synthetic.py` | `table6_synthetic.csv` | the separable control |
| `run_validation.py` | `validation_montecarlo.csv` | analytic vs. simulated probabilities |
| `run_diagnosis.py` | `table7_feature_diagnosis.csv` | why transfer fails on one graph |
| `run_comparison.py` | `table8_space_comparison.csv` | words of state to 10% error, against prediction-free methods |
| `run_simulation.py` | `table9_simulation.csv` | the estimator run for real: error vs. number of instances |
| `run_wedge.py` | `table10_wedge_predictions.csv` | predictions on the wedge estimator |
| `run_mechanism.py` | `table11_normalizer_inflation.csv` | which normalizer a skewed weighting inflates |
| `run_profile.py` | `table12_*`, `table13_*` | the higher-order triadic profile, exact and from a stream |
| `run_ranking.py` | `table14_*`, `table15_*` | ranking vertices by higher-order closure, exact and from a stream |
| `run_hardness.py` | `table16_hard_instances.csv` | our bound on the instances behind the known lower bounds |
| `run_learner_ablation.py` | `table17_*`, `table18_*` | ridge vs. gradient-boosted trees, and CIs for the copy-density rule |

The first run downloads ten hypergraphs from Benson's temporal higher-order collection into
`data/` (about 200 MB, via `gdown`; Google Drive throttles, so the loader retries). Expect
20–60 minutes end to end, dominated by `congress-bills`.

`results/` already holds the CSVs behind the submitted tables, so the paper's numbers can be
checked without re-running anything.

## What the code measures, and how

**Exact, not simulated.** Speedups are exact sums of per-copy path probabilities over the
sampling distribution, so comparisons carry no Monte-Carlo noise. `run_validation.py`
checks them against the actual sampler; agreement is within sampling error at 40k draws.

**Two mechanisms, kept apart.** The prediction-augmented sampler differs from the baseline
in two independent ways: the closing vertex is drawn against a *local* normalizer, and the
draws are *weighted*. Setting the weights to zero isolates the first. Every table reports

* `loc_x` — localization alone, over the prediction-free baseline;
* `perfect_over_loc` — what a perfect predictor adds on top;
* `permuted_over_loc` — the same weight multiset shuffled across edges (50 seeds);
* `structural_share` — the fraction of the predictor's gain the shuffle does not reproduce.

Only the second is a statement about prediction. Reporting the total would conflate the two.

**Per-edge counts without enumeration.** `t(e)` comes from a bitset identity — common
neighbours minus the hyperedge-covered part — so it is exact even on graphs with 4.5×10⁷
copies. Only the triple list used to sum path probabilities is subsampled on the two largest
graphs, and since every predictor is scored on the same sample, ratios are unaffected.

## Headline results

Ten hypergraphs, five domains, `m` from 1.8×10³ to 4.2×10⁵.

* **No exponent is available.** `alpha_H / kappa` ∈ [0.600, 0.836]; the degeneracy of the
  copy-bearing subgraph equals the degeneracy of the whole graph on eight of ten. The
  bracket `ceil(kappa_H/2) <= alpha_H <= kappa_H <= kappa` then pins the ratio above ½ by
  arithmetic, which is why the published band was never going to be beaten.
* **Localization is a constant.** 4.2×–9.5×, fitted growth exponent in `m` of +0.02. On the
  separable synthetic family the same quantity grows as `m^0.500`.
* **A perfect predictor adds 1.04×–4.08×** on top, and how much is predictable: the gain
  decays with copy density as `(#H/m)^-0.84` (r = −0.86).
* **The permutation control reverses.** For ordinary triangles a shuffled weight
  distribution retains ~91% of the benefit; here it retains none (structural share ≈ 1.00,
  permutation p = 0.0196 on every graph). For cross-hyperedge four-cycles, where there is no
  localization term at all, the structural share is 0.9995.
* **A cheap ridge model** over higher-order features recovers 0.47 of the achievable gain
  (95% CI [0.34, 0.60], 90 ordered pairs), against 0.28 for the degree-and-core features
  standard for ordinary triangles.
* **The min-degree heuristic does not transfer**: mean recovery −0.06 for triangles, though
  0.31 for four-cycles.

## Two checks worth running on your own data

```python
d = diagnostic_crossK3(hg)
d['kappa_copy_over_kappa']   # ~1 means no exponent gain is available; O(m) to compute
hg.n_cross_total / hg.m      # copy density; predicts whether a predictor is worth training
```

## Layout

```
crosscount/      core.py        degeneracy, exact oracle-width by max-flow, the bracket
                 hypergraph.py  CrossHG: projection, cross-triangle structure, features
                 crossc4.py     cross-hyperedge four-cycles
                 predictors.py  perfect / permuted / min-degree / ridge
                 evaluate.py    the localization-prediction split, Monte-Carlo check
                 bench.py       sample complexity, competing methods, estimator simulation
                 diagnosis.py   why transfer succeeds or fails on a given graph
                 profile.py     the higher-order triadic profile and its estimation
                 ranking.py     per-vertex closure scores and their recovery from a stream
                 hardness.py    known hard instances, measured in our parameters
                 datasets.py    ARB and SNAP loaders
                 synthetic.py   the control families, including AG(2,q)
experiments/     one script per table
notebooks/       the exploratory notebooks, in the order they were written
results/         the CSVs behind the submitted tables
paper/           the manuscript source
```

## Related repositories

* [`predcount`](https://github.com/Pedyi/predcount) — the prediction-augmented estimator for
  general patterns (*Array* 32, 101199, 2026).
* [`localcount`](https://github.com/Pedyi/localcount) — where the speedup comes from for
  ordinary triangles.

## Citation

```bibtex
@article{asadzadeh2026crosscount,
  author  = {Asadzadeh, Pedram},
  title   = {Predictions Buy Constants, Not Exponents: Streaming Higher-Order
             Triadic Closure and the Ceiling on Learned Oracles},
  journal = {Manuscript under review},
  year    = {2026}
}
```

Datasets are from Benson, Abebe, Schaub, Jadbabaie and Kleinberg, *Simplicial closure and
higher-order link prediction*, PNAS 115(48), 2018. Please cite them if you use the data.

## License

MIT.
