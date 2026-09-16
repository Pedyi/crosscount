"""crosscount --- prediction-augmented streaming counting of higher-order triadic closure.

Companion code for *Predictions Buy Constants, Not Exponents: Streaming Higher-Order
Triadic Closure and the Ceiling on Learned Oracles*.

Quick start
-----------
    from crosscount import CrossHG, evaluate, cliques_plus_cross

    hg = CrossHG(cliques_plus_cross(6, 40, 240))
    print(evaluate(hg, n_perm=50))
"""
from .core import (bits_to_idx, popcount, canon, build_adj, degeneracy,
                   degeneracy_arrays, oracle_width, bracket)
from .hypergraph import CrossHG, diagnostic_crossK3
from .crossc4 import CrossC4
from .predictors import (ridge_fit, ridge_predict_weights, gbt_fit,
                         gbt_predict_weights, cluster_bootstrap_ci, FEATURE_BLOCKS)
from .evaluate import evaluate, monte_carlo_check
from .diagnosis import feature_diagnosis
from .bench import (second_moment, weighted_cost, baseline_cost, wedge_cost,
                    edge_local_cost, heavy_storage_cost, simulate_estimator,
                    samples_for_error, wedge_weighted_cost, wedge_report,
                    normalizer_inflation)
from .profile import exact_profile, simulate_profile, domain_separation
from .ranking import (vertex_scores_exact, simulate_vertex_scores, precision_at_k,
                      rank_correlation)
from .hardness import hubs_graph, bera_chakrabarti, instance_report
from .synthetic import (friendship_plus_decoy, affine_plane_incidence,
                        steiner_plus_butterflies, cliques_plus_cross,
                        incidence_edges, random_uniform_hypergraph)

__version__ = '1.0.0'

__all__ = [
    'CrossHG', 'CrossC4', 'diagnostic_crossK3',
    'evaluate', 'monte_carlo_check', 'feature_diagnosis',
    'ridge_fit', 'ridge_predict_weights', 'gbt_fit', 'gbt_predict_weights',
    'cluster_bootstrap_ci',
    'FEATURE_BLOCKS',
    'oracle_width', 'degeneracy', 'degeneracy_arrays', 'bracket',
    'second_moment', 'weighted_cost', 'baseline_cost', 'wedge_cost',
    'edge_local_cost', 'heavy_storage_cost', 'simulate_estimator', 'samples_for_error',
    'wedge_weighted_cost', 'wedge_report', 'normalizer_inflation',
    'exact_profile', 'simulate_profile', 'domain_separation',
    'vertex_scores_exact', 'simulate_vertex_scores', 'precision_at_k', 'rank_correlation',
    'hubs_graph', 'bera_chakrabarti', 'instance_report',
    'bits_to_idx', 'popcount', 'canon', 'build_adj',
    'friendship_plus_decoy', 'affine_plane_incidence', 'steiner_plus_butterflies',
    'cliques_plus_cross', 'incidence_edges', 'random_uniform_hypergraph',
]
