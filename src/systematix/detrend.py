"""Detrenders: a common interface over plug-in filters and linear nuisance models.

Every detrender first divides each quarter by its median, giving relative
flux ``r = F / median_q(F) - 1``. Then:

* ``mode="subtract"`` (mode (a) in the design doc) estimates a trend and
  returns ``1 + r - trend``, optionally sigma clipped.
* ``mode="marginalize"`` (linear detrenders only; mode (c)) returns
  ``1 + r`` untouched, plus the design matrix ``M`` and a Gaussian prior on
  its weights, for a likelihood that integrates them out (e.g. spotgp's
  ``GPSolver``). ``prior_sigma = inf`` means a flat prior. Per-quarter offset
  columns are always included, so the plug-in median normalization does not
  matter there.

Linear detrenders fit their weights by generalized least squares with the
Gaussian prior, i.e. ridge regression when ``prior_sigma`` is finite.

Trends are subtracted in relative flux, not divided out; for percent-level
signals the difference is second order.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .observation import Observation

__all__ = [
    "DetrendResult", "Detrender", "LinearDetrender",
    "QuarterMedian", "RollingMedian", "LkFlatten", "Poly", "Ramps", "CBV", "ARC2",
    "DETRENDERS", "make_detrender", "normalize_quarters", "segment_poly_design",
]


@dataclass
class DetrendResult:
    time: np.ndarray
    flux: np.ndarray            # relative flux, ~1
    flux_err: np.ndarray
    quarter: np.ndarray
    mask: np.ndarray            # over the input cadences; True = kept
    design: np.ndarray | None = None
    prior_mu: np.ndarray | None = None
    prior_sigma: np.ndarray | None = None
    trend: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def full(self, values: np.ndarray | None = None) -> np.ndarray:
        """``values`` (default ``flux``) on the input cadences, NaN where dropped."""
        values = self.flux if values is None else values
        out = np.full(self.mask.shape, np.nan)
        out[self.mask] = values
        return out

    def binned(self, dt: float) -> "DetrendResult":
        """Inverse-variance average onto bins of ``dt`` days within each quarter.

        Design-matrix rows are averaged with the same weights, so a linear
        nuisance model stays exact after binning. ``mask`` then refers to the
        bins, not the input cadences.
        """
        from .sources import bin_weights

        ids, w, nb = bin_weights(self.time, self.quarter, self.flux_err, dt)
        sw = np.bincount(ids, weights=w, minlength=nb)

        def avg(v):
            return np.bincount(ids, weights=w * v, minlength=nb) / sw

        first = np.full(nb, len(self.time), dtype=np.int64)
        np.minimum.at(first, ids, np.arange(len(self.time)))
        design = None
        if self.design is not None:
            design = np.column_stack([avg(self.design[:, j]) for j in range(self.design.shape[1])])
        return DetrendResult(
            time=avg(self.time), flux=avg(self.flux), flux_err=1.0 / np.sqrt(sw),
            quarter=self.quarter[first], mask=np.ones(nb, dtype=bool), design=design,
            prior_mu=self.prior_mu, prior_sigma=self.prior_sigma,
            trend=None if self.trend is None else avg(self.trend),
            meta={**self.meta, "binned_dt": dt},
        )

    def gp_kwargs(self) -> dict:
        """``offsets`` and ``offset_prior`` for spotgp's ``GPSolver`` (marginalize mode).

        ``GPSolver(data, model, **res.gp_kwargs())`` integrates the nuisance
        columns out of the likelihood; ``prior_sigma = inf`` columns are flat.
        """
        if self.design is None:
            raise ValueError("only results from mode='marginalize' carry a design matrix")
        return {"offsets": self.design, "offset_prior": (self.prior_mu, self.prior_sigma)}


def normalize_quarters(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    """Relative flux ``F / median_q(F) - 1`` and its uncertainty, per quarter."""
    rel = np.full(len(obs), np.nan)
    err = np.full(len(obs), np.nan)
    for q, m in obs.quarter_masks().items():
        med = np.nanmedian(obs.flux[m])
        if np.isfinite(med) and med != 0:
            rel[m] = obs.flux[m] / med - 1.0
            err[m] = obs.flux_err[m] / abs(med)
    return rel, err


def _robust_clip(x: np.ndarray, keep: np.ndarray, k: float, iters: int = 3) -> np.ndarray:
    keep = keep.copy()
    for _ in range(iters):
        med = np.median(x[keep])
        mad = 1.4826 * np.median(np.abs(x[keep] - med))
        if mad == 0:
            break
        new = keep & (np.abs(x - med) <= k * mad)
        if new.sum() == keep.sum():
            break
        keep = new
    return keep


def _std_clip(x: np.ndarray, keep: np.ndarray, k: float) -> np.ndarray:
    """One pass of mean +/- k std (as in spotgp's scripts/injection_recovery.py)."""
    mean, std = np.mean(x[keep]), np.std(x[keep])
    return keep & (x >= mean - k * std) & (x <= mean + k * std)


class Detrender:
    """Base class. Subclasses implement :meth:`_trend` (and/or :meth:`_design`)."""

    name = "base"
    linear = False

    def __init__(self, clip: float | None = None, clip_method: str = "mad"):
        if clip_method not in ("mad", "std"):
            raise ValueError("clip_method must be 'mad' (iterative, median/MAD) or 'std' (one pass, mean/std)")
        self.clip = clip
        self.clip_method = clip_method

    def params(self) -> dict:
        return {"clip": self.clip, "clip_method": None if self.clip_method == "mad" else self.clip_method}

    @property
    def label(self) -> str:
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.3g}"
            if isinstance(v, tuple):
                return "(" + ", ".join(fmt(x) for x in v) + ")"
            return str(v)

        p = ", ".join(f"{k}={fmt(v)}" for k, v in self.params().items() if v is not None)
        return f"{self.name}({p})" if p else self.name

    def __repr__(self) -> str:
        return self.label

    def __call__(self, obs: Observation, mode: str = "subtract") -> DetrendResult:
        if mode not in ("subtract", "marginalize"):
            raise ValueError("mode must be 'subtract' or 'marginalize'")
        if mode == "marginalize" and not self.linear:
            raise ValueError(f"{self.name} is not linear and cannot be marginalized")
        rel, err = normalize_quarters(obs)
        good = np.isfinite(rel) & np.isfinite(err) & (err > 0)
        meta = {"detrender": self.name, "label": self.label, "mode": mode, **self.params()}

        if mode == "marginalize":
            M, mu, sigma = self._design(obs)
            M, mu, sigma = _drop_empty(M[good], mu, sigma)
            return DetrendResult(obs.time[good], 1.0 + rel[good], err[good], obs.quarter[good], good,
                                 design=M, prior_mu=mu, prior_sigma=sigma, meta=meta)

        self._extra_meta = {}
        trend = np.full(len(obs), np.nan)
        trend[good] = self._trend(obs, rel, err, good)[good]
        meta.update(self._extra_meta)       # per-call details a detrender chose to report
        out = rel - trend
        keep = good & np.isfinite(out)
        if self.clip:
            keep = (_robust_clip(out, keep, self.clip) if self.clip_method == "mad"
                    else _std_clip(out, keep, self.clip))
        return DetrendResult(obs.time[keep], 1.0 + out[keep], err[keep], obs.quarter[keep], keep,
                             trend=trend[keep], meta=meta)

    def _trend(self, obs, rel, err, good) -> np.ndarray:
        raise NotImplementedError

    def _design(self, obs):
        raise NotImplementedError


def _drop_empty(M, mu, sigma):
    """Drop design columns that are all zero on the kept cadences."""
    used = np.any(M != 0, axis=0)
    return M[:, used], mu[used], sigma[used]


def _gls(M, y, err, mu, sigma):
    """Weights minimising |(y - M w)/err|^2 + |(w - mu)/sigma|^2 (sigma=inf: flat)."""
    M, mu, sigma = _drop_empty(M, mu, sigma)
    A = M / err[:, None]
    b = y / err
    finite = np.isfinite(sigma)
    if finite.any():
        P = np.zeros((finite.sum(), M.shape[1]))
        P[np.arange(finite.sum()), np.flatnonzero(finite)] = 1.0 / sigma[finite]
        A = np.vstack([A, P])
        b = np.concatenate([b, mu[finite] / sigma[finite]])
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return M @ w


class LinearDetrender(Detrender):
    """Trend = M w with Gaussian prior on w; subclasses implement :meth:`_design`."""

    linear = True

    def _trend(self, obs, rel, err, good):
        M, mu, sigma = self._design(obs)
        trend = np.full(len(obs), np.nan)
        trend[good] = _gls(M[good], rel[good], err[good], mu, sigma)
        return trend


# ---------------------------------------------------------------- designs

def segment_poly_design(time: np.ndarray, labels: np.ndarray, order: int = 0) -> np.ndarray:
    """Per-segment polynomial columns: an indicator plus centred, scaled powers.

    Same construction as ``spotgp.offsets.offset_design``.
    """
    time = np.asarray(time, dtype=float)
    labels = np.asarray(labels)
    cols = []
    for s in np.unique(labels):
        m = labels == s
        cols.append(m.astype(float))
        if order > 0:
            xs = time[m] - time[m].mean()
            scale = np.abs(xs).max() or 1.0
            for p in range(1, order + 1):
                col = np.zeros(len(time))
                col[m] = (xs / scale) ** p
                cols.append(col)
    return np.column_stack(cols)


def _flat(n):
    return np.zeros(n), np.full(n, np.inf)


# ---------------------------------------------------------------- detrenders

class QuarterMedian(Detrender):
    """Divide each quarter by its median and nothing else (what ``stitch()`` does)."""

    name = "quarter_median"

    def _trend(self, obs, rel, err, good):
        return np.zeros(len(obs))


class RollingMedian(Detrender):
    """Running median over ``window`` days, within each quarter."""

    name = "rolling_median"

    def __init__(self, window: float = 10.0, clip: float | None = None, clip_method: str = "mad"):
        super().__init__(clip, clip_method)
        self.window = float(window)

    def params(self):
        return {"window": self.window, **super().params()}

    def _trend(self, obs, rel, err, good):
        trend = np.full(len(obs), np.nan)
        for q, m in obs.quarter_masks().items():
            idx = np.flatnonzero(m & good)
            t, y = obs.time[idx], rel[idx]
            lo = np.searchsorted(t, t - self.window / 2, side="left")
            hi = np.searchsorted(t, t + self.window / 2, side="right")
            trend[idx] = [np.median(y[a:b]) for a, b in zip(lo, hi)]
        return trend


class LkFlatten(Detrender):
    """lightkurve's ``flatten`` (iterative Savitzky-Golay) with a window in days."""

    name = "lk_flatten"

    def __init__(self, window: float = 2.0, polyorder: int = 2, break_tolerance: int = 5,
                 niters: int = 3, sigma: float = 3.0, clip: float | None = None, clip_method: str = "mad"):
        super().__init__(clip, clip_method)
        self.window, self.polyorder = float(window), int(polyorder)
        self.break_tolerance, self.niters, self.sigma = break_tolerance, niters, sigma

    def params(self):
        return {"window": self.window, "polyorder": self.polyorder, **super().params()}

    def _trend(self, obs, rel, err, good):
        import lightkurve as lk

        trend = np.full(len(obs), np.nan)
        for q, m in obs.quarter_masks().items():
            idx = np.flatnonzero(m & good)
            if idx.size < self.polyorder + 3:
                continue
            n = idx.size
            dt = np.median(np.diff(obs.time[idx]))
            wl = int(round(self.window / dt))
            wl = max(min(wl, n if n % 2 else n - 1), self.polyorder + 2)
            wl += 1 - wl % 2                                   # savgol needs an odd window
            lc = lk.LightCurve(time=obs.time[idx], flux=1.0 + rel[idx], flux_err=err[idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, tr = lc.flatten(window_length=wl, polyorder=self.polyorder, return_trend=True,
                                   break_tolerance=self.break_tolerance, niters=self.niters, sigma=self.sigma)
            trend[idx] = np.asarray(tr.flux.value, dtype=float) - 1.0
        return trend


class Poly(LinearDetrender):
    """Per-quarter polynomial of degree ``order`` (order 0 = free offsets only)."""

    name = "poly"

    def __init__(self, order: int = 2, clip: float | None = None, clip_method: str = "mad"):
        super().__init__(clip, clip_method)
        self.order = int(order)

    def params(self):
        return {"order": self.order, **super().params()}

    def _design(self, obs):
        M = segment_poly_design(obs.time, obs.quarter, self.order)
        return (M, *_flat(M.shape[1]))


class Ramps(LinearDetrender):
    """Exponential thermal-recovery ramps after each gap, plus a per-quarter polynomial.

    One column per (segment start, timescale): ``exp(-(t - t0) / tau)`` from
    each quarter start and each gap longer than ``min_gap`` days, until the
    next segment starts. Ramp amplitudes get a Gaussian prior of width
    ``prior_sigma`` (relative flux; None = flat).
    """

    name = "ramps"

    def __init__(self, taus=(0.5, 2.0), min_gap: float = 0.5, order: int = 1,
                 prior_sigma: float | None = None, clip: float | None = None, clip_method: str = "mad"):
        super().__init__(clip, clip_method)
        self.taus = tuple(float(t) for t in np.atleast_1d(taus))
        self.min_gap, self.order, self.prior_sigma = float(min_gap), int(order), prior_sigma

    def params(self):
        return {"taus": self.taus, "min_gap": self.min_gap, "order": self.order,
                "prior_sigma": self.prior_sigma, **super().params()}

    def _design(self, obs):
        base = segment_poly_design(obs.time, obs.quarter, self.order)
        starts = obs.segment_starts(self.min_gap)
        ends = np.r_[starts[1:], len(obs)]
        cols = []
        for s, e in zip(starts, ends):
            dt = obs.time[s:e] - obs.time[s]
            for tau in self.taus:
                col = np.zeros(len(obs))
                col[s:e] = np.exp(-dt / tau)
                cols.append(col)
        R = np.column_stack(cols) if cols else np.zeros((len(obs), 0))
        mu = np.zeros(base.shape[1] + R.shape[1])
        sigma = np.r_[np.full(base.shape[1], np.inf),
                      np.full(R.shape[1], np.inf if self.prior_sigma is None else self.prior_sigma)]
        return np.hstack([base, R]), mu, sigma


class _CBVMatrix:
    """CBV loading shared by :class:`CBV` and :class:`ARC2`.

    Injections reuse the same cadence many times, so CBV files are read once
    and the aligned matrix is cached per cadence. ``loader(quarter, module,
    output)`` overrides the MAST loader (tests).
    """

    def _cbv_matrix(self, obs: Observation, n_cbv: int) -> np.ndarray:
        from .cbv import cbv_design, load_cbv_quarter

        memo = self.__dict__.setdefault("_cbv_memo", {})
        key = (n_cbv, tuple(obs.quarters), len(obs), hash(obs.cadenceno.tobytes()))
        if key not in memo:
            loader = self.loader
            if loader is None:
                files = self.__dict__.setdefault("_cbv_files", {})

                def loader(q, mod, out):
                    if (q, mod, out) not in files:
                        files[(q, mod, out)] = load_cbv_quarter(q, mod, out, self.cache)
                    return files[(q, mod, out)]
            memo[key] = cbv_design(obs, n_cbv, loader=loader)
        return memo[key]


class CBV(_CBVMatrix, LinearDetrender):
    """Kepler DR25 CBVs per quarter (plus a per-quarter polynomial of ``order``).

    ``prior_sigma`` is the Gaussian prior width on each CBV weight in relative
    flux per unit-norm vector (None = flat, i.e. plain least squares, which
    overfits). A trend of rms ``s`` over N cadences needs weights of about
    ``s * sqrt(N)``. ``loader`` overrides the MAST loader (tests).
    """

    name = "cbv"

    def __init__(self, n_cbv: int = 8, prior_sigma: float | None = None, order: int = 0,
                 loader=None, cache: str | None = None, clip: float | None = None, clip_method: str = "mad"):
        super().__init__(clip, clip_method)
        self.n_cbv, self.prior_sigma, self.order = int(n_cbv), prior_sigma, int(order)
        self.loader, self.cache = loader, cache

    def params(self):
        return {"n_cbv": self.n_cbv, "prior_sigma": self.prior_sigma, "order": self.order,
                **super().params()}

    def _design(self, obs):
        V = self._cbv_matrix(obs, self.n_cbv)
        base = segment_poly_design(obs.time, obs.quarter, self.order)
        sigma = np.r_[np.full(base.shape[1], np.inf),
                      np.full(V.shape[1], np.inf if self.prior_sigma is None else self.prior_sigma)]
        return np.hstack([base, V]), np.zeros(sigma.size), sigma


class ARC2(_CBVMatrix, Detrender):
    """ARC2 CBV correction (Aigrain et al. 2017): ARD-shrinkage fit, adaptive CBV count.

    Per quarter, fits the first 1..``n_max`` CBVs with variational-Bayes ARD
    regression and keeps the number ARC2's range/noise rule selects (see
    :mod:`systematix.arc2`). Nonlinear, so subtract mode only. The
    selected count per quarter is reported in ``result.meta["n_cbv_selected"]``.

    ``backend="native"`` uses this package's MIT reimplementation;
    ``backend="oxksc"`` calls the original OxKeplerSC code, which must be
    importable or cloned at ``$OXKEPLERSC_PATH`` (or ``oxksc_path``).
    ARC2's jump correction is not included.
    """

    name = "arc2"

    def __init__(self, n_max: int = 8, backend: str = "native", loader=None, cache: str | None = None,
                 oxksc_path: str | None = None, clip: float | None = None, clip_method: str = "mad"):
        super().__init__(clip, clip_method)
        if backend not in ("native", "oxksc"):
            raise ValueError("backend must be 'native' or 'oxksc'")
        self.n_max, self.backend = int(n_max), backend
        self.loader, self.cache, self.oxksc_path = loader, cache, oxksc_path

    def params(self):
        return {"n_max": self.n_max, "backend": None if self.backend == "native" else self.backend,
                **super().params()}

    def _trend(self, obs, rel, err, good):
        from .arc2 import select_n_cbv

        if self.backend == "oxksc":
            from ._oxksc import load_oxksc_cbv, scipy_aliases
            oxksc = load_oxksc_cbv(self.oxksc_path)
        V = self._cbv_matrix(obs, self.n_max)
        trend = np.full(len(obs), np.nan)
        selected = {}
        for j, q in enumerate(obs.quarters):
            rows = np.flatnonzero((obs.quarter == q) & good)
            y = 1.0 + rel[rows]
            B = V[rows, j * self.n_max:(j + 1) * self.n_max]
            if self.backend == "native":
                fit = select_n_cbv(y, B, self.n_max)
                corrected, selected[q] = fit.corrected, fit.n_cbv
            else:
                with scipy_aliases():
                    (n_sel, corrected, _), _ = oxksc.sel_nb(y, B, nBmax=self.n_max)
                selected[q] = int(n_sel)
            trend[rows] = y - corrected
        self._extra_meta = {"n_cbv_selected": selected}
        return trend


DETRENDERS = {
    cls.name: cls for cls in (QuarterMedian, RollingMedian, LkFlatten, Poly, Ramps, CBV, ARC2)
}


def make_detrender(spec: str | dict) -> Detrender:
    """Build from a name or a config dict, e.g. ``{"name": "poly", "order": 2}``."""
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = spec.pop("name")
    spec.pop("mode", None)
    if name not in DETRENDERS:
        raise KeyError(f"unknown detrender {name!r}; known: {sorted(DETRENDERS)}")
    return DETRENDERS[name](**spec)
