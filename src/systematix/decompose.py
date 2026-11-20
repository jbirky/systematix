"""Decomposition of a quiet star's SAP flux into the Paper V systematics blocks.

The design matrix ``M`` has four blocks, all per quarter:

- ``offsets``: one indicator per segment (quarters split at pointing steps);
- ``polynomial``: centred, scaled linear and quadratic terms per segment;
- ``ramps``: ``exp(-(t - t0) / tau)`` after each quarter start and each gap
  longer than ``MIN_GAP``, for ``tau`` in ``TAUS``;
- ``cbv``: the first ``N_CBV`` DR25 CBVs of the star's output.

:func:`fit_flat` fits all blocks together under flat priors, with a jitter set
so the reduced chi-square is 1. :func:`keyed_weights` re-expresses one
quarter's weights on keys that mean the same thing for every star on an output:

- ``("offset", k)``: segment k's offset relative to the quarter's first segment
  (so a level jump at a step is an offset difference plus polynomial edge
  values, and both are drawn together);
- ``("poly", k, p)``;
- ``("ramp", t0, tau)``, with t0 snapped to the output's canonical gap-end times;
- ``("cbv", j)``.

Ported from the Phase 4 pilot (``validation/phase4_pilot/pilot_lib.py``), which
the design doc's steps 2-5 were tuned on. One addition: a segment shorter than
``MIN_POLY_CADENCES`` keeps its offset but gets no polynomial columns, which
would otherwise be degenerate with the offset.
"""

from __future__ import annotations

import numpy as np

from .cbv import cbv_design
from .detrend import Ramps, segment_poly_design
from .observation import Observation
from .steps import segment_labels

__all__ = ["POLY_ORDER", "TAUS", "MIN_GAP", "N_CBV", "T0_JOIN", "BLOCKS", "KIND_ORDER",
           "design_blocks", "fit_flat", "canonical_t0", "weight_columns", "keyed_weights",
           "key_str", "parse_key", "sort_keys"]

POLY_ORDER = 2
TAUS = (0.5, 2.0)          # d: ramp timescales
MIN_GAP = 0.5              # d: gaps longer than this start a new ramp
N_CBV = 8
T0_JOIN = 0.3              # d: ramp start times this close are the same gap in different stars
MIN_POLY_CADENCES = 48     # segments shorter than this (1 d of long cadence) get no polynomial terms
BLOCKS = ("offsets", "polynomial", "ramps", "cbv")
KIND_ORDER = ("offset", "poly", "ramp", "cbv")


def design_blocks(obs: Observation, steps, n_cbv: int = N_CBV, cbv_loader=None,
                  poly_order: int = POLY_ORDER) -> dict:
    """Blocks of M for ``obs`` (good cadences only), with column metadata.

    ``steps`` are the pointing-step times to split segments at (usually an
    output's consensus steps). ``cbv_loader(quarter, module, output)`` replaces
    the MAST CBV loader (see :func:`systematix.cbv.cbv_design`). ``poly_order``
    is the highest power of time per segment (``poly_order`` columns each; 0
    leaves the polynomial block empty). The Phase 4 donor models use the
    default, ``POLY_ORDER``.
    """
    if poly_order < 0:
        raise ValueError(f"poly_order must be >= 0, got {poly_order}")
    seg = segment_labels(obs.time, obs.quarter, steps)
    P = segment_poly_design(obs.time, seg, poly_order)
    per = poly_order + 1
    off_cols = np.arange(0, P.shape[1], per)
    poly_cols = np.setdiff1d(np.arange(P.shape[1]), off_cols)
    useg = np.unique(seg)
    short = np.array([np.sum(seg == s) < MIN_POLY_CADENCES for s in useg])
    Pp = P[:, poly_cols].copy()
    for i in np.flatnonzero(short):
        Pp[:, i * poly_order:(i + 1) * poly_order] = 0.0     # dropped as all-zero columns by fit_flat
    nq = len(obs.quarters)
    R = Ramps(taus=TAUS, min_gap=MIN_GAP, order=0)._design(obs)[0][:, nq:]
    starts = obs.segment_starts(MIN_GAP)
    V = cbv_design(obs, n_cbv, loader=cbv_loader)
    return {
        "offsets": P[:, off_cols], "polynomial": Pp, "ramps": R, "cbv": V,
        "seg": seg, "seg_ids": useg,
        "poly_meta": [(int(s // 100), int(s), p) for s in useg for p in range(1, per)],
        "ramp_meta": [(int(obs.quarter[s]), round(float(obs.time[s]), 1), tau) for s in starts for tau in TAUS],
        "cbv_meta": [(q, j) for q in obs.quarters for j in range(n_cbv)],
    }


def fit_flat(rel, err, D: dict, iters: int = 5):
    """Flat-prior GLS of all blocks, with a jitter so the reduced chi-square is 1.

    Returns ``(weights, residual, jitter, cond)``. ``weights`` maps each block
    to ``(w, C)`` and ``"all"`` to the full vector and covariance (cross-block
    terms included). Columns that are zero on every cadence are dropped and get
    NaN weights. ``cond`` is the condition number of the normal matrix.
    """
    M = np.hstack([D[b] for b in BLOCKS])
    sizes = [D[b].shape[1] for b in BLOCKS]
    live = np.any(M != 0, axis=0)
    Ml = M[:, live]
    jit = 0.0
    for _ in range(iters):
        var = err ** 2 + jit ** 2
        A = (Ml.T / var) @ Ml
        C = np.linalg.inv(A)
        w = C @ (Ml.T @ (rel / var))
        res = rel - Ml @ w
        jit = np.sqrt(max(np.mean(res ** 2 - err ** 2), 0.0))
    wf = np.full(M.shape[1], np.nan)
    wf[live] = w
    Cf = np.full((M.shape[1], M.shape[1]), np.nan)
    Cf[np.ix_(live, live)] = C
    out, i0 = {}, 0
    for b, n in zip(BLOCKS, sizes):
        out[b] = (wf[i0:i0 + n], Cf[i0:i0 + n, i0:i0 + n])
        i0 += n
    out["all"] = (wf, Cf)
    return out, res, jit, float(np.linalg.cond(A))


def canonical_t0(t0_lists, join: float = T0_JOIN) -> np.ndarray:
    """Gap-end times shared across stars: all stars' ramp start times, clustered within ``join`` d."""
    t = np.sort(np.concatenate([np.asarray(x, dtype=float) for x in t0_lists] or [np.array([])]))
    if t.size == 0:
        return np.array([])
    return np.array([np.median(g) for g in np.split(t, np.flatnonzero(np.diff(t) > join) + 1)])


def weight_columns(D: dict, q: int, t0c) -> tuple[dict, int | None]:
    """({key: column of M}, reference offset column) for quarter ``q``; M stacks BLOCKS in order."""
    t0c = np.asarray(t0c, dtype=float)
    sizes = [D[b].shape[1] for b in BLOCKS]
    start = dict(zip(BLOCKS, np.cumsum([0] + sizes[:-1])))
    off = {int(s): start["offsets"] + i for i, s in enumerate(D["seg_ids"])}
    segs = [s for s in off if s // 100 == q]
    cols = {("offset", s % 100): off[s] for s in segs[1:]}
    for j, (qq, sid, p) in enumerate(D["poly_meta"]):
        if qq == q:
            cols[("poly", sid % 100, p)] = start["polynomial"] + j
    for j, (qq, t0, tau) in enumerate(D["ramp_meta"]):
        if qq == q and t0c.size:
            i = int(np.argmin(np.abs(t0c - t0)))
            if abs(t0c[i] - t0) <= T0_JOIN:
                cols[("ramp", round(float(t0c[i]), 1), tau)] = start["ramps"] + j
    for j, (qq, c) in enumerate(D["cbv_meta"]):
        if qq == q:
            cols[("cbv", c)] = start["cbv"] + j
    return cols, (off[segs[0]] if segs else None)


def keyed_weights(weights: dict, D: dict, q: int, t0c) -> tuple[list, np.ndarray, np.ndarray]:
    """(keys, w, C): quarter ``q``'s fitted weights on the keys of :func:`weight_columns`.

    Keys whose column was dropped by the fit (NaN weight) are left out.
    """
    wall, Call = weights["all"]
    cols, ref = weight_columns(D, q, t0c)
    keys = [k for k in cols if np.isfinite(wall[cols[k]])]
    T = np.zeros((len(keys), len(wall)))
    for r, k in enumerate(keys):
        T[r, cols[k]] = 1.0
        if k[0] == "offset":
            T[r, ref] -= 1.0
    return keys, T @ np.nan_to_num(wall), T @ np.nan_to_num(Call) @ T.T


def _sort_key(k):
    return (KIND_ORDER.index(k[0]),) + tuple(k[1:])


def key_str(k) -> str:
    """Storage form of a weight key, e.g. ``"ramp:200.3:0.5"``."""
    return ":".join(str(x) for x in k)


def parse_key(s: str) -> tuple:
    """Inverse of :func:`key_str`."""
    kind, *rest = s.split(":")
    if kind == "offset":
        return (kind, int(rest[0]))
    if kind == "poly":
        return (kind, int(rest[0]), int(rest[1]))
    if kind == "ramp":
        return (kind, float(rest[0]), float(rest[1]))
    if kind == "cbv":
        return (kind, int(rest[0]))
    raise ValueError(f"unknown weight key {s!r}")


def sort_keys(keys) -> list:
    """Keys in block order (offsets, polynomials, ramps, CBVs)."""
    return sorted(keys, key=_sort_key)
