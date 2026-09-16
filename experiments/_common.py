"""Shared setup for the experiment scripts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MAX_TRIPLES = 4_000_000

from crosscount.datasets import load_all, DOMAIN, PAPER_DATASETS   # noqa: E402
from crosscount import CrossHG                                     # noqa: E402


def build_all(names=None, max_triples=MAX_TRIPLES, verbose=True):
    """Fetch the datasets and build the cross-triangle structure for each."""
    hyper = load_all(names, cache_dir=DATA, verbose=verbose)
    hg = {}
    for name, h in hyper.items():
        if verbose:
            print(f'{name} ...')
        hg[name] = CrossHG(h, max_size=25, max_triples=max_triples, seed=0, verbose=verbose)
    return hyper, hg


def save(df, filename):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, filename)
    df.to_csv(path, index=False)
    print(f'wrote {path}')
    return path
