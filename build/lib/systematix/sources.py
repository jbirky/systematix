"""Systematics sources and the injection model.

A source produces a :class:`Systematics` realization on a real (or synthetic)
cadence. :func:`inject` combines it with a simulated stellar signal::

    F_obs(t) = scale(t) * [c_q * F_spot(t) + (1 - c_q)] + n(t)

where ``scale = L_q * m_q(t)`` is the per-quarter flux level times the
multiplicative systematics, ``c_q`` is the quarter's ``CROWDSAP`` (dilution by
neighbouring stars) and ``n`` is white noise of standard deviation ``sigma``.
``F_spot`` is the stellar signal normalized to 1.

Sources (IDs from the design doc):

* S0 :class:`NoSystematics` -- real or synthetic cadence and gaps only.
* S1 :class:`PDCRatio` -- the multiplicative systematics PDC removed from a
  donor star (``sap_flux / pdcsap_flux_rescaled``), plus white noise.
* S2 :class:`QuietSAP` -- a quiet donor's raw SAP flux, which already holds
  real noise, granulation and systematics PDC cannot model.
* :class:`DonorPool` -- S1 or S2 from a random donor per realization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation import Observation

__all__ = [
    "Systematics", "inject", "synthetic_cadence",
    "NoSystematics", "PDCRatio", "QuietSAP", "DonorPool",
]


@dataclass
class Systematics:
    """One systematics realization on a fixed cadence (see module docstring)."""

    template: Observation   # cadence, quarters, quality, cadenceno, quarter_meta
    scale: np.ndarray       # L_q * m_q(t)
    sigma: np.ndarray       # white-noise standard deviation to add
    flux_err: np.ndarray    # reported uncertainty of the injected flux
    dilution: bool = True
    name: str = ""
    donor: int | None = None

    def __len__(self) -> int:
        return len(self.template)

    @property
    def time(self) -> np.ndarray:
        return self.template.time

    def crowding(self) -> np.ndarray:
        return self.template.crowding() if self.dilution else np.ones(len(self))

    def draw_noise(self, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.standard_normal(len(self)) * self.sigma


def inject(sys: Systematics, f_spot=1.0, noise: np.ndarray | None = None, rng=None) -> Observation:
    """Observed flux for stellar signal ``f_spot`` (normalized to 1) under ``sys``.

    Pass the same ``noise`` array to compare runs with and without a signal
    (paired trials); otherwise noise is drawn from ``rng``.
    """
    f_spot = np.broadcast_to(np.asarray(f_spot, dtype=float), (len(sys),))
    if noise is None:
        noise = sys.draw_noise(rng)
    c = sys.crowding()
    flux = sys.scale * (c * f_spot + (1.0 - c)) + noise
    truth = {"f_spot": f_spot.copy(), "scale": sys.scale, "crowding": c,
             "noise": noise, "source": sys.name, "donor": sys.donor}
    return sys.template.replace(flux=flux, flux_err=sys.flux_err.copy(), truth=truth)


def _bin_index(time: np.ndarray, quarter: np.ndarray, dt: float) -> np.ndarray:
    """Bin id per cadence: bins of width ``dt`` that never straddle quarters."""
    ids = np.empty(time.size, dtype=np.int64)
    offset = 0
    for q in np.unique(quarter):
        m = quarter == q
        local = np.floor((time[m] - time[m].min()) / dt).astype(np.int64)
        _, inv = np.unique(local, return_inverse=True)
        ids[m] = inv + offset
        offset += inv.max() + 1
    return ids


def bin_weights(time, quarter, flux_err, dt):
    """(bin ids, inverse-variance weights, number of bins) for :func:`bin_systematics`."""
    ids = _bin_index(np.asarray(time), np.asarray(quarter), dt)
    w = 1.0 / np.asarray(flux_err, dtype=float) ** 2
    return ids, w, int(ids.max()) + 1


def bin_systematics(sys: Systematics, dt: float) -> Systematics:
    """Average a realization onto bins of ``dt`` days within each quarter.

    Weights are inverse variances from ``flux_err`` (as in spotgp's
    ``scripts/injection_recovery.py``). White noise ``sigma`` averages down
    accordingly, so injecting into the binned realization is equivalent to
    injecting a signal sampled at the bin centres and then binning.
    """
    t = sys.template
    ids, w, nb = bin_weights(t.time, t.quarter, sys.flux_err, dt)
    sw = np.bincount(ids, weights=w, minlength=nb)

    def avg(v):
        return np.bincount(ids, weights=w * v, minlength=nb) / sw

    first = np.full(nb, len(t), dtype=np.int64)
    np.minimum.at(first, ids, np.arange(len(t)))
    quality = np.zeros(nb, dtype=np.int64)
    np.bitwise_or.at(quality, ids, t.quality)
    template = Observation(
        time=avg(t.time), flux=avg(t.flux), flux_err=1.0 / np.sqrt(sw),
        quarter=t.quarter[first], quality=quality, cadenceno=t.cadenceno[first],
        quarter_meta={q: dict(m) for q, m in t.quarter_meta.items()}, kic=t.kic,
    )
    sigma = np.sqrt(np.bincount(ids, weights=w ** 2 * sys.sigma ** 2, minlength=nb)) / sw
    return Systematics(template, scale=avg(sys.scale), sigma=sigma, flux_err=1.0 / np.sqrt(sw),
                       dilution=sys.dilution, name=f"{sys.name}|bin{dt:g}", donor=sys.donor)


def synthetic_cadence(
    n_quarters: int = 2,
    quarter_days: float = 90.0,
    dt: float = 0.02043,
    gap_every: float = 31.0,
    gap_days: float = 1.0,
    sigma: float = 1e-4,
    start_quarter: int = 1,
    seed: int | None = 0,
) -> Observation:
    """A Kepler-like long-cadence template with monthly downlink gaps, flux = 1.

    Lets S0 and the tests run without downloading anything. Quarters are
    separated by 1-day gaps.
    """
    rng = np.random.default_rng(seed)
    times, quarters = [], []
    t0 = 0.0
    for i in range(n_quarters):
        t = t0 + np.arange(0.0, quarter_days, dt)
        keep = np.ones(t.size, dtype=bool)
        for start in np.arange(t0 + gap_every, t0 + quarter_days - gap_days, gap_every):
            keep &= ~((t >= start) & (t < start + gap_days * rng.uniform(0.8, 1.2)))
        times.append(t[keep])
        quarters.append(np.full(keep.sum(), start_quarter + i))
        t0 += quarter_days + 1.0
    time = np.concatenate(times)
    cadenceno = np.round((time - time[0]) / dt).astype(np.int64)
    return Observation(
        time=time, flux=np.ones_like(time), flux_err=np.full_like(time, sigma),
        quarter=np.concatenate(quarters), cadenceno=cadenceno,
        quarter_meta={start_quarter + i: {"CROWDSAP": 1.0, "FLFRCSAP": 1.0}
                      for i in range(n_quarters)},
    )


class NoSystematics:
    """S0: the template's cadence and gaps, no systematics, no dilution.

    ``scale`` is 1 (or each quarter's median template flux if ``offsets``),
    and the white noise is the template's relative uncertainty unless
    ``sigma`` is given.
    """

    name = "S0:none"

    def __init__(self, template: Observation, sigma: float | None = None, offsets: bool = False):
        self.template = template
        self.sigma = sigma
        self.offsets = offsets

    def realize(self, rng=None) -> Systematics:
        t = self.template
        level = np.ones(len(t))
        rel_err = np.empty(len(t))
        for q, m in t.quarter_masks().items():
            med = np.nanmedian(t.flux[m])
            rel_err[m] = t.flux_err[m] / med
            if self.offsets:
                level[m] = med
        sigma = level * (rel_err if self.sigma is None else self.sigma)
        return Systematics(t, scale=level, sigma=sigma, flux_err=sigma.copy(),
                           dilution=False, name=self.name, donor=t.kic)


def _load_donor(kic_or_lc, quarters, download_dir):
    if isinstance(kic_or_lc, Observation):
        raise TypeError("pass a KIC id or a get_systematics() light curve, not an Observation")
    if hasattr(kic_or_lc, "meta"):
        return kic_or_lc
    from .core import get_systematics

    return get_systematics(kic_or_lc, quarters=quarters, normalize=False, download_dir=download_dir)


def _col(lc, name):
    v = lc[name]
    v = getattr(v, "value", v)
    return np.ma.filled(np.ma.asarray(v, dtype=float), np.nan)


class PDCRatio:
    """S1: multiplicative systematics PDC removed from a donor, plus white noise.

    ``scale = L_q * sap_flux / pdcsap_flux_rescaled``, with ``L_q`` the
    donor's median PDC flux (SAP scale) in each quarter, and white noise at
    the donor's ``sap_flux_err``. The ratio is nearly noise free because SAP
    and PDC share the same photons.

    Caveat: PDC builds its correction from CBVs, so CBV detrenders recover S1
    almost perfectly. Use quiet donors, or the ratio carries the long-period
    stellar signal PDC removed.
    """

    name = "S1:pdc_ratio"

    def __init__(self, donor, quarters=None, dilution: bool = True, download_dir=None):
        self._donor, self.quarters, self.dilution, self.download_dir = donor, quarters, dilution, download_dir
        self._sys = None

    def realize(self, rng=None) -> Systematics:
        if self._sys is None:
            lc = _load_donor(self._donor, self.quarters, self.download_dir)
            t = Observation.from_systematics(lc)
            ratio = _col(lc, "ratio")
            pdc = _col(lc, "pdcsap_flux_rescaled")
            level = np.empty(len(t))
            for q, m in t.quarter_masks().items():
                level[m] = np.nanmedian(pdc[m])
            good = np.isfinite(ratio) & np.isfinite(t.flux_err)
            t = t.select(good)
            scale = (level * ratio)[good]
            self._sys = Systematics(t, scale=scale, sigma=t.flux_err.copy(), flux_err=t.flux_err.copy(),
                                    dilution=self.dilution, name=self.name, donor=t.kic)
        return self._sys


class QuietSAP:
    """S2: a quiet donor's SAP flux as the full noise-plus-systematics model.

    ``scale`` is the donor's SAP flux itself and no extra noise is added. The
    donor's own contamination fraction ``CROWDSAP`` dilutes the injected
    signal. "Quiet" means no detected period, not no variability.
    """

    name = "S2:quiet_sap"

    def __init__(self, donor, quarters=None, dilution: bool = True, download_dir=None):
        self._donor, self.quarters, self.dilution, self.download_dir = donor, quarters, dilution, download_dir
        self._sys = None

    def realize(self, rng=None) -> Systematics:
        if self._sys is None:
            lc = _load_donor(self._donor, self.quarters, self.download_dir)
            t = Observation.from_systematics(lc, flux_column="sap_flux")
            good = np.isfinite(t.flux) & np.isfinite(t.flux_err)
            t = t.select(good)
            self._sys = Systematics(t, scale=t.flux.copy(), sigma=np.zeros(len(t)), flux_err=t.flux_err.copy(),
                                    dilution=self.dilution, name=self.name, donor=t.kic)
        return self._sys


class DonorPool:
    """S1 or S2 from a donor drawn at random for each realization.

    ``max_pdcvar`` rejects donors whose mean ``PDCVAR`` header value (PDC's
    variability estimate) exceeds it; up to ``max_tries`` donors are tried.
    Loaded donors are kept in memory.
    """

    def __init__(self, kics, kind: str = "pdc_ratio", quarters=None, max_pdcvar: float | None = None,
                 max_tries: int = 10, dilution: bool = True, download_dir=None):
        kinds = {"pdc_ratio": PDCRatio, "quiet_sap": QuietSAP}
        if kind not in kinds:
            raise ValueError(f"kind must be one of {sorted(kinds)}")
        self.kics = np.asarray(kics, dtype=np.int64)
        self.cls, self.kind = kinds[kind], kind
        self.quarters, self.max_pdcvar, self.max_tries = quarters, max_pdcvar, max_tries
        self.dilution, self.download_dir = dilution, download_dir
        self._loaded: dict[int, object] = {}
        self.name = f"pool:{kind}"

    def realize(self, rng=None) -> Systematics:
        rng = np.random.default_rng(rng)
        for _ in range(self.max_tries):
            kic = int(rng.choice(self.kics))
            if kic not in self._loaded:
                try:
                    self._loaded[kic] = self.cls(kic, self.quarters, self.dilution, self.download_dir)
                    sys = self._loaded[kic].realize()
                except LookupError:
                    self._loaded[kic] = None
                    continue
            src = self._loaded[kic]
            if src is None:
                continue
            sys = src.realize()
            if self.max_pdcvar is not None:
                pdcvar = np.nanmean(sys.template.header("PDCVAR"))
                if not np.isfinite(pdcvar) or pdcvar > self.max_pdcvar:
                    continue
            return sys
        raise RuntimeError(f"No acceptable donor found in {self.max_tries} draws")
