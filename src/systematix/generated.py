"""S3: generated systematics, drawn from the Paper V weights of quiet donors.

:class:`WeightModel` is the joint distribution of one (quarter, CCD channel)'s
weights over its donors (design doc step 3): the donors' keyed weights from
:mod:`systematix.decompose`, their mean, and a method-of-moments covariance
(sample covariance minus the donors' mean fit covariance, projected to
positive semi-definite). It also keeps each donor's Kepler magnitude and
residual noise: fitted jitter, its white part, and the red excess of the
residual spectrum (:mod:`systematix.rednoise`).

:class:`Generated` (step 4) builds the Paper V design matrix on a template
cadence and draws ``m = 1 + M w``. By default ``w`` is one donor's fitted
weight vector in every quarter ("resample"), from the 50 donors closest in Kp
to the template, and the noise is that donor's: white noise at the template's
errors plus the donor's white jitter, and red noise with the donor's residual
spectrum. Whole donors are resampled because the weights are heavy-tailed: in
the Phase 4 pilot a Gaussian with the right covariance matched the mean power
but was 1.3-1.4x too strong at the median. "gaussian" draws from the moment
model instead.

Model files are written by the Phase 4 production run
(``validation/phase4_production/weights.py``) as ``<dir>/Qqq/chCC.h5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from . import rednoise as rn
from .decompose import BLOCKS, design_blocks, key_str, parse_key, weight_columns
from .observation import Observation
from .sources import Systematics

__all__ = ["WeightModel", "Generated", "load_weight_model", "DRAWS", "KP_MATCH"]

DRAWS = ("resample", "gaussian")
KP_MATCH = 50            # default number of Kp-nearest donors for resampling
BLOCK_KIND = {"offsets": "offset", "polynomial": "poly", "ramps": "ramp", "cbv": "cbv"}


def _factor(S):
    """(L with L L^T = S clipped to PSD, clipped share of the trace)."""
    S = 0.5 * (S + S.T)
    lam, U = np.linalg.eigh(S)
    pos = np.clip(lam, 0, None)
    return U * np.sqrt(pos), float(-lam[lam < 0].sum() / max(pos.sum(), 1e-300))


def _rows(a, m):
    return None if a is None else a[m]


@dataclass
class WeightModel:
    """Weights of one (quarter, channel) over its donors, their moment model, and the donors' noise."""

    quarter: int
    channel: int
    keys: list
    kic: np.ndarray            # (n,) donors
    W: np.ndarray              # (n, k) fitted weights
    Ve: np.ndarray             # (n, k, k) their fit covariances
    jitter: np.ndarray         # (n,) fitted jitter (relative flux)
    steps: np.ndarray          # consensus pointing steps of the output
    t0c: np.ndarray            # canonical ramp start times of the output
    kepmag: np.ndarray | None = None        # (n,) donors' Kepler magnitudes (for Kp-matched draws)
    red: np.ndarray | None = None           # (n, nbins) red excess of the residual spectrum (rednoise)
    white_jitter: np.ndarray | None = None  # (n,) white part of the jitter (used with red noise)
    red_edges: np.ndarray | None = None     # (nbins + 1,) bin edges of ``red`` in c/d
    mean: np.ndarray = field(init=False)
    cov: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    clipped: float = field(init=False)

    def __post_init__(self):
        self.kic = np.asarray(self.kic, dtype=np.int64)
        self.W = np.asarray(self.W, dtype=float)
        self.Ve = np.asarray(self.Ve, dtype=float)
        self.jitter = np.asarray(self.jitter, dtype=float)
        self.steps = np.asarray(self.steps, dtype=float)
        self.t0c = np.asarray(self.t0c, dtype=float)
        n = len(self.kic)
        self.kepmag = (np.full(n, np.nan) if self.kepmag is None
                       else np.broadcast_to(np.asarray(self.kepmag, dtype=float), (n,)).copy())
        if n < 2:
            raise ValueError(f"Q{self.quarter} ch{self.channel}: need at least 2 donors, got {n}")
        if self.red is not None:
            self.red_edges = np.asarray(rn.RED_EDGES if self.red_edges is None else self.red_edges, dtype=float)
            self.red = np.asarray(self.red, dtype=float).reshape(n, len(self.red_edges) - 1)
            if self.white_jitter is None:
                raise ValueError("red spectra need white_jitter (rednoise.white_jitter)")
            self.white_jitter = np.asarray(self.white_jitter, dtype=float).reshape(n)
        self.mean = self.W.mean(axis=0)
        S = np.cov(self.W, rowvar=False).reshape(len(self.keys), len(self.keys)) - self.Ve.mean(axis=0)
        self.L, self.clipped = _factor(S)
        self.cov = self.L @ self.L.T

    # ------------------------------------------------------------ construction
    @classmethod
    def estimate(cls, quarter, channel, keys, kic, W, C, jitter, steps, t0c, min_frac: float = 0.9, kepmag=None,
                 red=None, white_jitter=None, red_edges=None):
        """Model from one output's decomposition, keeping keys that ``min_frac`` of the donors have.

        Donors missing any kept key drop out. ``W`` (n, K) holds NaN where a donor
        lacks a key; ``C`` (n, K, K) is the fit covariance. ``red`` and
        ``white_jitter`` come from :func:`systematix.rednoise.residual_spectra`
        and :func:`~systematix.rednoise.white_jitter` on the donors' residuals.
        """
        W, C = np.asarray(W, float), np.asarray(C, float)
        present = np.isfinite(W)
        keep = present.mean(axis=0) >= min_frac
        rows = np.all(present[:, keep], axis=1)
        j = np.flatnonzero(keep)
        sub = lambda a: None if a is None else np.asarray(a, float)[rows]           # noqa: E731
        return cls(int(quarter), int(channel), [keys[i] for i in j], np.asarray(kic)[rows],
                   W[np.ix_(rows, j)], C[rows][:, j][:, :, j], np.asarray(jitter)[rows], steps, t0c,
                   sub(kepmag), sub(red), sub(white_jitter), red_edges)

    def subset(self, kics) -> "WeightModel":
        """The same keys re-estimated on the donors in ``kics`` (for cross-validation)."""
        m = np.isin(self.kic, np.asarray(kics))
        return WeightModel(self.quarter, self.channel, list(self.keys), self.kic[m], self.W[m], self.Ve[m],
                           self.jitter[m], self.steps, self.t0c, self.kepmag[m], _rows(self.red, m),
                           _rows(self.white_jitter, m), self.red_edges)

    def nearest_kp(self, kepmag: float, n: int, among=None) -> np.ndarray:
        """Row indices of the ``n`` donors closest in Kp to ``kepmag`` (optionally only rows ``among``)."""
        rows = np.arange(len(self.kic)) if among is None else np.asarray(among, dtype=int)
        if np.all(np.isnan(self.kepmag[rows])):
            raise ValueError(f"Q{self.quarter} ch{self.channel}: donor Kp unknown, cannot match")
        d = np.abs(np.nan_to_num(self.kepmag[rows], nan=np.inf) - kepmag)
        return rows[np.argsort(d, kind="stable")[:n]]

    # ------------------------------------------------------------ queries
    @property
    def kind(self) -> np.ndarray:
        return np.array([k[0] for k in self.keys])

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def ramp_sd(self) -> float:
        d = np.diag(self.cov)[self.kind == "ramp"]
        return float(np.sqrt(d.mean())) if d.size else 0.0

    @property
    def has_red(self) -> bool:
        return self.red is not None

    def draw(self, rng, how: str = "resample", donor: int | None = None):
        """(weights, jitter): one donor's (``resample``; ``donor`` picks it) or a Gaussian draw."""
        if how == "resample":
            i = rng.integers(len(self.kic)) if donor is None else int(np.flatnonzero(self.kic == donor)[0])
            return self.W[i], float(self.jitter[i])
        if how == "gaussian":
            return self.mean + self.L @ rng.standard_normal(self.L.shape[1]), float(rng.choice(self.jitter))
        raise ValueError(f"draw must be one of {DRAWS}, got {how!r}")

    # ------------------------------------------------------------ I/O
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            h5.attrs.update(quarter=self.quarter, channel=self.channel, clipped=self.clipped)
            h5.create_dataset("keys", data=np.array([key_str(k) for k in self.keys], dtype="S"))
            for name in ("kic", "W", "jitter", "steps", "t0c", "kepmag", "mean", "cov"):
                h5.create_dataset(name, data=getattr(self, name))
            h5.create_dataset("Ve", data=self.Ve.astype(np.float32), compression="gzip", shuffle=True)
            if self.red is not None:
                h5.create_dataset("red", data=self.red.astype(np.float32), compression="gzip", shuffle=True)
                h5.create_dataset("white_jitter", data=self.white_jitter)
                h5.create_dataset("red_edges", data=self.red_edges)

    @classmethod
    def load(cls, path) -> "WeightModel":
        with h5py.File(path, "r") as h5:
            keys = [parse_key(s.decode()) for s in h5["keys"][:]]
            opt = {name: (h5[name][:] if name in h5 else None) for name in ("kepmag", "red", "white_jitter",
                                                                             "red_edges")}
            if opt["red"] is not None:
                opt["red"] = opt["red"].astype(float)
            return cls(int(h5.attrs["quarter"]), int(h5.attrs["channel"]), keys, h5["kic"][:], h5["W"][:],
                       h5["Ve"][:].astype(float), h5["jitter"][:], h5["steps"][:], h5["t0c"][:], **opt)


def load_weight_model(model_dir, quarter: int, channel: int) -> WeightModel:
    """The model file ``<model_dir>/Qqq/chCC.h5``."""
    return WeightModel.load(Path(model_dir) / f"Q{int(quarter):02d}" / f"ch{int(channel):02d}.h5")


class Generated:
    """S3: systematics drawn from donor weights on a template cadence.

    ``template`` supplies the cadence, gaps, quality flags, crowding and flux
    errors (a real star's good cadences, with ``MODULE``/``OUTPUT``/``CHANNEL``
    and ``KEPMAG`` in ``quarter_meta``). ``models`` maps each of its quarters
    to a :class:`WeightModel`, or is a model directory (the channel is read
    from ``quarter_meta``).

    ``draw="resample"`` (the default) takes one donor's weights in every
    quarter (a donor with models in all the template's quarters when there is
    one, otherwise one donor per quarter) and that donor's per-quarter noise.
    ``"gaussian"`` draws from the moment model, with the noise of a random
    donor.

    ``kp_match`` (resample only) draws from the N donors closest in Kepler
    magnitude to the template (``KEPMAG`` in ``quarter_meta``). The default,
    ``"auto"``, is :data:`KP_MATCH` (50) for resampling and no matching for the
    Gaussian draw; ``None`` pools the model's donors of every Kp. Matching is
    the default because the systematics in relative flux grow with Kp: in the
    full Phase 4 gate (1,460 output-quarters), donors pooled over Kp 12-15 made
    S3 too strong on templates brighter than Kp 14 (median power ratio 0.56)
    and too weak on fainter ones (1.38), although the pooled gate passed. The
    50 nearest donors passed in 98.8% and 99.9% of the cells (ratios 0.91 and
    0.99).

    ``red_noise`` (default on) adds red noise with the same donor's residual
    spectrum (:mod:`systematix.rednoise`) and uses the white part of its
    jitter, so the total noise variance is unchanged and only its spectrum
    moves. The red noise belongs to the realization's noise: each
    :meth:`Systematics.draw_noise` call draws it afresh, and it is not part of
    ``scale``. It needs models with residual spectra (``WeightModel.red``).

    ``blocks`` switches design blocks on (``offsets``, ``polynomial``,
    ``ramps``, ``cbv``), and ``jitter`` the extra white noise, so effects can
    be tested one at a time. Template ramps at gaps the donors do not share get
    independent draws at the model's pooled ramp sd. The realization is
    normalized per quarter (median 1), as real data are.
    """

    name = "S3:generated"

    def __init__(self, template: Observation, models, draw: str = "resample", blocks=BLOCKS,
                 jitter: bool = True, dilution: bool = True, kp_match: int | None | str = "auto",
                 red_noise: bool = True, cbv_loader=None, design: dict | None = None):
        if draw not in DRAWS:
            raise ValueError(f"draw must be one of {DRAWS}, got {draw!r}")
        if isinstance(kp_match, str):
            if kp_match != "auto":
                raise ValueError(f"kp_match must be 'auto', an integer or None, got {kp_match!r}")
            kp_match = KP_MATCH if draw == "resample" else None
        if kp_match is not None and draw != "resample":
            raise ValueError("kp_match works with draw='resample' only")
        unknown = set(blocks) - set(BLOCKS)
        if unknown:
            raise ValueError(f"unknown blocks {sorted(unknown)}; choose from {BLOCKS}")
        self.template, self.how, self.blocks = template, draw, tuple(blocks)
        self.use_jitter, self.dilution, self.red_noise = bool(jitter), bool(dilution), bool(red_noise)
        quarters = template.quarters
        if not isinstance(models, dict):
            model_dir = models
            models = {}
            for q in quarters:
                ch = template.quarter_meta.get(q, {}).get("CHANNEL")
                if ch is None:
                    raise ValueError(f"quarter {q}: CHANNEL missing from quarter_meta")
                models[q] = load_weight_model(model_dir, q, int(ch))
        missing = [q for q in quarters if q not in models]
        if missing:
            raise ValueError(f"no weight model for quarters {missing}")
        self.models = {q: models[q] for q in quarters}
        if self.red_noise:
            lacking = [q for q in quarters if not self.models[q].has_red]
            if lacking:
                raise ValueError(f"the weight models for quarters {lacking} have no residual spectra; rebuild them"
                                 " (validation/phase4_production/weights.py) or pass red_noise=False")
        steps = np.concatenate([self.models[q].steps for q in quarters])
        D = design if design is not None else design_blocks(template, steps, cbv_loader=cbv_loader)
        M = np.hstack([D[b] for b in BLOCKS])
        on = {BLOCK_KIND[b] for b in self.blocks}
        self.X, self.extra = {}, {}
        for q in quarters:
            mdl = self.models[q]
            cols, _ = weight_columns(D, q, mdl.t0c)
            X = np.zeros((len(template), len(mdl.keys)))
            for i, key in enumerate(mdl.keys):
                if key in cols and key[0] in on:
                    X[:, i] = M[:, cols[key]]
            self.X[q] = X
            extra = [cols[k] for k in cols if k[0] == "ramp" and k not in mdl.keys] if "ramp" in on else []
            self.extra[q] = M[:, extra]
        self.common = np.array(sorted(set.intersection(*(set(self.models[q].kic.tolist()) for q in quarters))),
                               dtype=np.int64)
        # donor pools: a common donor in every quarter when possible, else one per quarter
        self.kp_match = kp_match
        first = self.models[quarters[0]]
        if kp_match is None:
            self.pool_common = self.common
            self.pool_q = {q: np.arange(len(m.kic)) for q, m in self.models.items()}
        else:
            kp = template.quarter_meta.get(quarters[0], {}).get("KEPMAG")
            if kp is None or not np.isfinite(kp):
                raise ValueError("kp_match needs KEPMAG in the template's quarter_meta"
                                 " (kp_match=None pools donors of every Kp)")
            self.template_kp = float(kp)
            if self.common.size >= kp_match:
                rows = np.flatnonzero(np.isin(first.kic, self.common))
                self.pool_common = first.kic[first.nearest_kp(self.template_kp, kp_match, among=rows)]
            else:
                self.pool_common = np.array([], dtype=np.int64)
            self.pool_q = {q: m.nearest_kp(self.template_kp, kp_match) for q, m in self.models.items()}
        self.masks = template.quarter_masks()
        self.level = np.empty(len(template))
        for q, m in self.masks.items():
            self.level[m] = np.nanmedian(template.flux[m])
        self.grid = ({q: rn.cadence_grid(template.time[m], template.cadenceno[m]) for q, m in self.masks.items()}
                     if self.red_noise else {})

    # ------------------------------------------------------------ draws
    def _draw(self, rng):
        """({quarter: weights}, {quarter: model row whose noise is used}, donor KIC or None)."""
        W, rows = {}, {}
        donor = None
        if self.how == "resample" and self.pool_common.size:
            donor = int(self.pool_common[rng.integers(self.pool_common.size)])
        for q, mdl in self.models.items():
            if self.how == "resample":
                i = (int(np.flatnonzero(mdl.kic == donor)[0]) if donor is not None
                     else int(rng.choice(self.pool_q[q])))
                W[q] = mdl.W[i]
            else:
                W[q] = mdl.mean + mdl.L @ rng.standard_normal(mdl.L.shape[1])
                i = int(rng.integers(len(mdl.kic)))
            rows[q] = i
        return W, rows, donor

    def _jitter(self, q, i) -> float:
        mdl = self.models[q]
        return float(mdl.white_jitter[i] if self.red_noise else mdl.jitter[i])

    def _trend(self, rng):
        W, rows, donor = self._draw(rng)
        s = np.zeros(len(self.template))
        for q, w in W.items():
            s += self.X[q] @ w
            if self.extra[q].shape[1]:
                s += self.extra[q] @ rng.normal(0.0, self.models[q].ramp_sd, self.extra[q].shape[1])
        for q, m in self.masks.items():
            s[m] -= np.median(s[m])
        return s, rows, donor

    def draw_weights(self, rng):
        """({quarter: weights}, {quarter: white jitter}, donor KIC or None)."""
        W, rows, donor = self._draw(rng)
        return W, {q: self._jitter(q, i) for q, i in rows.items()}, donor

    def draw_trend(self, rng):
        """(relative systematics s with per-quarter median 0, {quarter: white jitter}, donor)."""
        s, rows, donor = self._trend(rng)
        return s, {q: self._jitter(q, i) for q, i in rows.items()}, donor

    def _red_draw(self, rows):
        """A function ``rng -> red noise`` (flux units) with the residual spectra of ``rows``."""
        spec = {q: (self.models[q].red[i], self.models[q].red_edges) for q, i in rows.items()}
        masks, grid, level, n = self.masks, self.grid, self.level, len(self.template)

        def draw(rng):
            out = np.zeros(n)
            for q, m in masks.items():
                idx, N, dt = grid[q]
                red, edges = spec[q]
                out[m] = rn.red_noise(red, idx, N, dt, rng, edges)
            return out * level

        return draw

    def realize(self, rng=None) -> Systematics:
        rng = np.random.default_rng(rng)
        s, rows, donor = self._trend(rng)
        t = self.template
        rel_err = t.flux_err / self.level
        if self.use_jitter:
            jit = np.empty(len(t))
            for q, m in self.masks.items():
                jit[m] = self._jitter(q, rows[q])
            rel_err = np.hypot(rel_err, jit)
        return Systematics(t, scale=self.level * (1.0 + s), sigma=rel_err * self.level,
                           flux_err=t.flux_err.copy(), dilution=self.dilution, name=self.name, donor=donor,
                           red=self._red_draw(rows) if self.red_noise else None)
