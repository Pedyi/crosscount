"""Dataset loaders.

The ten hypergraphs come from Benson et al.'s temporal higher-order collection
(https://www.cs.cornell.edu/~arb/data).  The archives are hosted on Google Drive, so plain
urlretrieve fails and gdown is used with the file ids below.  Each archive holds
`<name>-nverts.txt` and `<name>-simplices.txt`.
"""
import glob
import gzip
import os
import tarfile
import time
import urllib.request

__all__ = ['ARB_IDS', 'DOMAIN', 'fetch_arb', 'load_all', 'load_simplices',
           'load_hyperedges_lines', 'fetch', 'load_edge_list', 'load_temporal',
           'SNAP_STATIC', 'SNAP_TEMPORAL']

ARB_IDS = {
    'contact-primary-school': '1sBHSEIyvVKavAho524Ro4cKL66W6rn-t',
    'contact-high-school':    '1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu',
    'email-Enron':            '1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77',
    'email-Eu':               '1amLeVudLBDRglCXKlieg6HHE-vu81EVF',
    'NDC-classes':            '1tpDiP1c73O18gCYEx4OI7kx8V_IdYxLt',
    'NDC-substances':         '1mGOg0DMh46J2zQdimSXMde1pKNtfAdh8',
    'DAWN':                   '1wGwoG7oBWnNN7J9TEpjqNpODbsYfMxp4',
    'congress-bills':         '1gH1uJMZpn_SCJSRbORPH4JRQeevLwTyO',
    'tags-math-sx':           '1eDevpF6EZs19rLouNpiKGLIlFOLUfKKG',
    'tags-ask-ubuntu':        '1tb1ZJlXEJnlRkXpTuBZlOqqsFknWCkUV',
    # not used in the paper; n = 125k vertices, slow with the bitset engine
    'threads-ask-ubuntu':     '1ppdJ7CvF_aJ9GLJmVWgKpNL55YVH8MYh',
}

DOMAIN = {
    'contact-primary-school': 'contact', 'contact-high-school': 'contact',
    'email-Enron': 'email', 'email-Eu': 'email',
    'NDC-classes': 'drugs', 'NDC-substances': 'drugs', 'DAWN': 'drugs',
    'congress-bills': 'legislation',
    'tags-math-sx': 'tags', 'tags-ask-ubuntu': 'tags',
    'threads-ask-ubuntu': 'threads',
}

PAPER_DATASETS = [
    'contact-primary-school', 'contact-high-school', 'email-Enron', 'email-Eu',
    'NDC-classes', 'NDC-substances', 'DAWN', 'congress-bills',
    'tags-math-sx', 'tags-ask-ubuntu',
]

SNAP_STATIC = {
    'ca-GrQc':  'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
    'ca-HepTh': 'https://snap.stanford.edu/data/ca-HepTh.txt.gz',
    'facebook': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
}

SNAP_TEMPORAL = {
    'CollegeMsg':             'https://snap.stanford.edu/data/CollegeMsg.txt.gz',
    'email-Eu-core-temporal': 'https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz',
    'sx-mathoverflow':        'https://snap.stanford.edu/data/sx-mathoverflow.txt.gz',
}


def load_simplices(nverts_path, simplices_path):
    """ARB format: one simplex size per line, plus a flat stream of vertex ids."""
    sizes = [int(x) for x in open(nverts_path).read().split()]
    flat = [int(x) for x in open(simplices_path).read().split()]
    out, i = [], 0
    for s in sizes:
        out.append(tuple(flat[i:i + s]))
        i += s
    return out


def load_hyperedges_lines(path):
    """One hyperedge per line, whitespace- or comma-separated vertex ids."""
    op = gzip.open if path.endswith('.gz') else open
    out = []
    with op(path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            out.append(tuple(line.replace(',', ' ').split()))
    return out


def _extract(tgz, dest='.'):
    with tarfile.open(tgz) as t:
        try:
            t.extractall(dest, filter='data')
        except TypeError:                  # Python < 3.12
            t.extractall(dest)


def fetch_arb(name, cache_dir='.', tries=3):
    """Download and extract one ARB hypergraph. Returns the hyperedge list, or None.

    Google Drive throttles; we retry and fall back to the fuzzy URL form.
    """
    import gdown
    os.makedirs(cache_dir, exist_ok=True)
    tgz = os.path.join(cache_dir, f'{name}.tar.gz')
    for k in range(tries):
        try:
            if not os.path.exists(tgz) or os.path.getsize(tgz) < 1000:
                if not gdown.download(id=ARB_IDS[name], output=tgz, quiet=True):
                    gdown.download(url=f'https://drive.google.com/uc?id={ARB_IDS[name]}',
                                   output=tgz, quiet=True, fuzzy=True)
            _extract(tgz, cache_dir)
            nv = glob.glob(os.path.join(cache_dir, '**', f'{name}-nverts.txt'), recursive=True)
            sp = glob.glob(os.path.join(cache_dir, '**', f'{name}-simplices.txt'), recursive=True)
            return load_simplices(nv[0], sp[0])
        except Exception as e:
            print(f'  {name}: attempt {k + 1}/{tries} failed ({type(e).__name__})')
            if os.path.exists(tgz) and os.path.getsize(tgz) < 1000:
                os.remove(tgz)
            time.sleep(5)
    print(f'  {name}: giving up; fetch it by hand from '
          f'https://drive.google.com/uc?id={ARB_IDS[name]}')
    return None


def load_all(names=None, cache_dir='data', max_size=25, verbose=True):
    """Fetch the paper's datasets and return {name: list of unique simplices}."""
    names = names or PAPER_DATASETS
    out = {}
    for name in names:
        h = fetch_arb(name, cache_dir=cache_dir)
        if h is None:
            continue
        uniq = list(dict.fromkeys(tuple(sorted(set(x))) for x in h))
        out[name] = [x for x in uniq if 2 <= len(x) <= max_size]
        if verbose:
            print(f'{name:24s} {len(out[name]):>7d} unique simplices')
    return out


# ------------------------------------------------------------------ SNAP helpers

def fetch(url, dest):
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
    return dest


def load_edge_list(path):
    op = gzip.open if path.endswith('.gz') else open
    E = []
    with op(path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            a, b = line.split()[:2]
            E.append((a, b))
    return E


def load_temporal(path):
    """SNAP temporal edge list 'u v t'. Returns rows sorted by timestamp."""
    op = gzip.open if path.endswith('.gz') else open
    rows = []
    with op(path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) < 3:
                continue
            rows.append((int(p[2]), int(p[0]), int(p[1])))
    rows.sort()
    return rows
