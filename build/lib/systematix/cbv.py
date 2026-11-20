"""Kepler DR25 cotrending basis vectors (CBVs), aligned to an Observation.

Each quarter has one ~30 MB FITS file on MAST holding the CBVs for all 84 CCD
outputs. Files are cached under ``$SYSTEMATIX_CACHE/cbv``. Only
single-scale CBVs are archived for Kepler (PDC itself used multi-scale MAP).

The vectors are zero-mean and roughly unit-norm over a quarter, so a fitted
coefficient of ``a`` on relative flux corresponds to a trend of rms about
``a / sqrt(N_quarter)``.
"""

from __future__ import annotations

import json
import re
import urllib.request

import numpy as np
from astropy.io import fits

from ._cache import cache_dir, download
from .observation import Observation

__all__ = ["CBV_BASE_URL", "cbv_filename", "load_cbv_quarter", "cbv_design"]

CBV_BASE_URL = "https://archive.stsci.edu/missions/kepler/cbv/"
N_CBV_MAX = 16


def cbv_filename(quarter: int, cache: str | None = None) -> str:
    """Name of the DR25 CBV file for ``quarter`` (the MAST listing is cached)."""
    listing = cache_dir("cbv", cache) / "listing.json"
    if listing.exists():
        names = json.loads(listing.read_text())
    else:
        with urllib.request.urlopen(CBV_BASE_URL, timeout=120) as resp:
            html = resp.read().decode()
        names = sorted(set(re.findall(r'href="(kplr\d+-q\d\d-d25_lcbv\.fits)"', html)))
        if not names:
            raise RuntimeError(f"No DR25 CBV files found at {CBV_BASE_URL}")
        listing.write_text(json.dumps(names))
    tag = f"-q{int(quarter):02d}-d25_lcbv.fits"
    match = [n for n in names if n.endswith(tag)]
    if len(match) != 1:
        raise LookupError(f"Expected one DR25 CBV file for quarter {quarter}, found {match}")
    return match[0]


def load_cbv_quarter(quarter: int, module: int, output: int, cache: str | None = None) -> dict:
    """CBVs for one quarter and CCD output.

    Returns ``{"cadenceno": (M,), "vectors": (M, 16), "gap": (M,)}``.
    """
    name = cbv_filename(quarter, cache)
    path = download(CBV_BASE_URL + name, cache_dir("cbv", cache) / name)
    ext = f"MODOUT_{int(module)}_{int(output)}"
    with fits.open(path, memmap=False) as hdul:
        data = hdul[ext].data
        vectors = np.column_stack([np.asarray(data[f"VECTOR_{i}"], dtype=float)
                                   for i in range(1, N_CBV_MAX + 1)])
        names = data.columns.names
        gap_col = "GAPFLAG" if "GAPFLAG" in names else ("GAP" if "GAP" in names else None)
        cadenceno = np.asarray(data["CADENCENO"], dtype=np.int64)
        gap = (np.asarray(data[gap_col], dtype=bool) if gap_col
               else np.zeros(cadenceno.size, dtype=bool))
        return {"cadenceno": cadenceno, "vectors": vectors, "gap": gap}


def cbv_design(obs: Observation, n_cbv: int = 8, loader=None, cache: str | None = None) -> np.ndarray:
    """Block-diagonal CBV design matrix for ``obs``: ``n_cbv`` columns per quarter.

    Rows are matched on ``cadenceno``. Cadences missing from the CBV table, or
    flagged as gaps there, get zero rows. ``loader(quarter, module, output)``
    replaces :func:`load_cbv_quarter` (used in tests).
    """
    if not 1 <= n_cbv <= N_CBV_MAX:
        raise ValueError(f"n_cbv must be in 1..{N_CBV_MAX}")
    loader = loader or (lambda q, m, o: load_cbv_quarter(q, m, o, cache))
    quarters = obs.quarters
    M = np.zeros((len(obs), n_cbv * len(quarters)))
    for j, q in enumerate(quarters):
        meta = obs.quarter_meta.get(q, {})
        module, output = meta.get("MODULE"), meta.get("OUTPUT")
        if module is None or output is None:
            raise ValueError(f"quarter {q}: MODULE/OUTPUT missing from quarter_meta")
        cbv = loader(q, int(module), int(output))
        rows = np.flatnonzero(obs.quarter == q)
        cad = obs.cadenceno[rows]
        idx = np.searchsorted(cbv["cadenceno"], cad)
        idx = np.clip(idx, 0, len(cbv["cadenceno"]) - 1)
        found = (cbv["cadenceno"][idx] == cad) & ~cbv["gap"][idx]
        block = np.where(found[:, None], cbv["vectors"][idx, :n_cbv], 0.0)
        M[rows, j * n_cbv:(j + 1) * n_cbv] = np.nan_to_num(block)
    return M
