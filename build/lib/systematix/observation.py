"""The container that systematics sources produce and detrenders consume."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Observation"]


@dataclass
class Observation:
    """A Kepler-like light curve on a real cadence, one or more quarters.

    ``flux`` is in SAP-like units (e-/s, with a different level per quarter)
    unless a source chose otherwise; detrenders normalize it themselves.
    ``quarter_meta`` maps each quarter to its FITS header values
    (``CROWDSAP``, ``FLFRCSAP``, ``MODULE``, ``OUTPUT``, ...).
    ``truth`` holds whatever the source and injection put in (the injected
    spot flux, the systematics realization, ...) for later scoring.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quarter: np.ndarray
    quality: np.ndarray | None = None
    cadenceno: np.ndarray | None = None
    quarter_meta: dict[int, dict] = field(default_factory=dict)
    kic: int | None = None
    truth: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        n = self.time.shape[0]
        self.flux = np.asarray(self.flux, dtype=float)
        self.flux_err = np.broadcast_to(np.asarray(self.flux_err, dtype=float), (n,)).copy()
        self.quarter = np.broadcast_to(np.asarray(self.quarter, dtype=int), (n,)).copy()
        self.quality = (np.zeros(n, dtype=np.int64) if self.quality is None
                        else np.asarray(self.quality, dtype=np.int64))
        self.cadenceno = (np.full(n, -1, dtype=np.int64) if self.cadenceno is None
                          else np.asarray(self.cadenceno, dtype=np.int64))
        for name in ("flux", "quality", "cadenceno"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected ({n},)")
        if n > 1 and np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing")

    def __len__(self) -> int:
        return self.time.shape[0]

    @property
    def quarters(self) -> list[int]:
        return sorted(int(q) for q in np.unique(self.quarter))

    def quarter_masks(self) -> dict[int, np.ndarray]:
        return {q: self.quarter == q for q in self.quarters}

    def header(self, key: str, default: float = np.nan) -> np.ndarray:
        """Per-cadence array of a per-quarter header value."""
        out = np.full(len(self), default, dtype=float)
        for q, m in self.quarter_masks().items():
            val = self.quarter_meta.get(q, {}).get(key)
            if val is not None:
                out[m] = float(val)
        return out

    def crowding(self) -> np.ndarray:
        """Per-cadence ``CROWDSAP`` (1 where unknown)."""
        return self.header("CROWDSAP", default=1.0)

    def segment_starts(self, min_gap: float = 0.5) -> np.ndarray:
        """Indices where data resume: each quarter's first cadence and every gap > ``min_gap`` days."""
        if len(self) == 0:
            return np.array([], dtype=int)
        new_q = np.r_[True, self.quarter[1:] != self.quarter[:-1]]
        gap = np.r_[True, np.diff(self.time) > min_gap]
        return np.flatnonzero(new_q | gap)

    def replace(self, **changes) -> "Observation":
        """Copy with some fields changed (arrays not passed are copied)."""
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for k, v in fields.items():
            if isinstance(v, np.ndarray):
                fields[k] = v.copy()
        fields["quarter_meta"] = {q: dict(m) for q, m in self.quarter_meta.items()}
        fields["truth"] = dict(self.truth)
        fields.update(changes)
        return Observation(**fields)

    def select(self, mask: np.ndarray) -> "Observation":
        """Subset of cadences (``truth`` arrays of full length are subset too)."""
        mask = np.asarray(mask, dtype=bool)
        truth = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape == (len(self),) else v)
                 for k, v in self.truth.items()}
        return self.replace(
            time=self.time[mask], flux=self.flux[mask], flux_err=self.flux_err[mask],
            quarter=self.quarter[mask], quality=self.quality[mask],
            cadenceno=self.cadenceno[mask], truth=truth,
        )

    @classmethod
    def from_systematics(cls, lc, flux_column: str = "sap_flux") -> "Observation":
        """Build from a :func:`systematix.get_systematics` light curve.

        ``flux_column`` picks which column becomes ``flux`` (SAP by default).
        """
        def col(name):
            v = lc[name]
            v = getattr(v, "value", v)
            return np.ma.filled(np.ma.asarray(v, dtype=float), np.nan)

        qmeta = lc.meta.get("QUARTER_META") or {}
        return cls(
            time=np.asarray(lc.time.value, dtype=float),
            flux=col(flux_column),
            flux_err=col("sap_flux_err"),
            quarter=col("quarter").astype(int),
            quality=col("quality").astype(np.int64),
            cadenceno=col("cadenceno").astype(np.int64),
            quarter_meta={int(q): dict(m) for q, m in qmeta.items()},
            kic=lc.meta.get("KEPLERID"),
        )
