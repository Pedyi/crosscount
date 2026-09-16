# results

The CSVs behind the submitted tables, so the paper's numbers can be checked without
re-running anything. Regenerate any of them with the matching script in `../experiments`.

| file | script | contents |
|---|---|---|
| `table1_diagnostic.csv` | `run_diagnostic.py` | exact `alpha_H`, `kappa`, `kappa_H` on the ten hypergraph projections |
| `table1b_object_class_scan.csv` | notebook 01 | the wider diagnostic scan: incidence graphs and 75 temporal windows |
| `table2_main.csv` | `run_main.py` | localization, perfect predictor, permutation null, min-degree |
| `table3_learned_pairs.csv` | `run_learned.py` | every ordered train/test pair, three feature blocks, three penalties |
| `table3_learned_summary.csv` | `run_learned.py` | capture per block with bootstrap CIs |
| `table4_crossc4.csv` | `run_crossc4.py` | cross-hyperedge four-cycles on the four graphs within budget |
| `table5_cap_sensitivity.csv` | `run_robustness.py` | `congress-bills` at four dataset sizes |
| `table5_triple_sensitivity.csv` | `run_robustness.py` | three triple-subsampling fractions and seeds |
| `table6_synthetic.csv` | `run_synthetic.py` | the separable family, where localization grows as `m^0.5` |
| `table7_feature_diagnosis.csv` | `run_diagnosis.py` | target spread and in-sample fit per graph |
| `validation_montecarlo.csv` | `run_validation.py` | analytic vs. simulated success probabilities |

Column conventions are described in the docstring of `crosscount/evaluate.py`. In short:
`loc_x` is localization over the prediction-free baseline, and everything named
`*_over_loc` is measured on top of it, so only those columns are statements about
prediction.
