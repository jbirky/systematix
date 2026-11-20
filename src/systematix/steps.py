"""Kepler pointing steps from ``POS_CORR1/2``, and the segments they define.

The detector is the one in Paper V's ``decomposition_model.py`` (moved here so
the paper and the generated source S3 share one implementation): a step is a
``POS_CORR`` jump of more than ``POINTING_STEP`` pixels between adjacent good
cadences, or across a data gap, where the median positions over
``STEP_WINDOW`` days on either side are compared (the pointing can change
while no data are taken). Jumps during the first day of a quarter (settling)
are ignored, and jumps within half a day merge.

Across a group of stars on one CCD output, :func:`consensus_steps` keeps the
step times found in at least half of them, so every star gets the same
segments.
"""

from __future__ import annotations

import numpy as np

__all__ = ["POINTING_STEP", "ADJACENT", "STEP_WINDOW", "DEFAULT_BITMASK",
           "pointing_steps", "consensus_steps", "segment_labels"]

POINTING_STEP = 0.02      # px: POS_CORR jump that counts as a step
ADJACENT = 0.1            # d: cadences closer than this are compared directly; wider spacings are gaps
STEP_WINDOW = 0.5         # d: across a gap, median positions over this much data on each side
SETTLE = 1.0              # d: ignore jumps this soon after the quarter's first good cadence
MERGE = 0.5               # d: steps closer than this merge
DEFAULT_BITMASK = 1130799  # lightkurve KeplerQualityFlags.DEFAULT_BITMASK


def pointing_steps(time, pos_corr1, pos_corr2, quality=None, bitmask: int = DEFAULT_BITMASK,
                   threshold: float = POINTING_STEP) -> list[float]:
    """Step times [d] in one quarter of one star.

    Takes the raw (unmasked) cadences; cadences with non-finite positions or
    any ``bitmask`` quality bit are dropped first. Each step is placed midway
    between the two cadences it separates.
    """
    t = np.asarray(time, dtype=float)
    pos = np.column_stack([np.asarray(pos_corr1, dtype=float), np.asarray(pos_corr2, dtype=float)])
    ok = np.isfinite(t) & np.all(np.isfinite(pos), axis=1)
    if quality is not None:
        ok &= (np.asarray(quality, dtype=np.int64) & bitmask) == 0
    tt, pp = t[ok], pos[ok]
    if len(tt) < 3:
        return []
    o = np.argsort(tt)
    tt, pp = tt[o], pp[o]
    dt = np.diff(tt)
    jump = np.hypot(*np.diff(pp, axis=0).T)
    for i in np.flatnonzero(dt >= ADJACENT):
        before = pp[(tt > tt[i] - STEP_WINDOW) & (tt <= tt[i])]
        after = pp[(tt >= tt[i + 1]) & (tt < tt[i + 1] + STEP_WINDOW)]
        jump[i] = np.hypot(*(np.median(after, axis=0) - np.median(before, axis=0)))
    hit = (jump > threshold) & (tt[:-1] > tt[0] + SETTLE)
    steps: list[float] = []
    for i in np.flatnonzero(hit):
        ts = 0.5 * (tt[i] + tt[i + 1])
        if not steps or ts - steps[-1] > MERGE:
            steps.append(float(ts))
    return steps


def consensus_steps(step_lists, frac: float = 0.5, join: float = 0.5) -> list[float]:
    """Step times found in at least ``frac`` of the stars, clustered within ``join`` days.

    ``step_lists`` holds one list of step times per star (empty lists count).
    Each kept cluster is represented by its median time.
    """
    step_lists = list(step_lists)
    times = [np.asarray(s, dtype=float) for s in step_lists if len(s)]
    if not times:
        return []
    allt = np.sort(np.concatenate(times))
    groups = np.split(allt, np.flatnonzero(np.diff(allt) > join) + 1)
    return [float(np.median(g)) for g in groups if len(g) >= frac * len(step_lists)]


def segment_labels(time, quarter, steps) -> np.ndarray:
    """Segment label per cadence, ``100 * quarter + k``: quarters split at the steps.

    ``k`` counts the steps that fall inside the same quarter's time span and
    before the cadence, so it restarts at 0 in every quarter. A label then means
    the same segment for every star given the same steps, whether a star is fit
    one quarter at a time or several together.
    """
    time = np.asarray(time, dtype=float)
    quarter = np.asarray(quarter, dtype=int)
    steps = np.sort(np.asarray(steps, dtype=float))
    out = quarter * 100
    for q in np.unique(quarter):
        m = quarter == q
        tq = time[m]
        sq = steps[(steps > tq.min()) & (steps < tq.max())]
        out[m] += np.searchsorted(sq, tq)
    return out
