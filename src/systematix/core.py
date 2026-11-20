"""Kepler systematics = SAP_FLUX minus the PDC-corrected flux.

PDC does two things to SAP_FLUX:

1. **Cotrending** (PDC-MAP / msMAP): it subtracts a systematics model built from
   cotrending basis vectors. This is what we want to recover.
2. **Crowding and flux-fraction correction**: it removes the contaminating flux
   and rescales for the target flux that falls outside the aperture
   (Kinemuchi et al. 2012, PASP 124, 963)::

       PDCSAP = (F_cot - (1 - CROWDSAP) * median(F_cot)) / FLFRCSAP

   where ``F_cot`` is the cotrended flux on the SAP scale.

A plain ``SAP - PDCSAP`` mixes the two steps. The crowding term adds a large
constant offset, and the ``1/FLFRCSAP`` rescaling leaks a fraction
``(1 - 1/FLFRCSAP)`` of the star's own variability into the result. By default
we undo step 2 first, which puts PDCSAP back on the SAP scale::

    F_cot = FLFRCSAP * PDCSAP + (1 - CROWDSAP) * FLFRCSAP * median(PDCSAP) / CROWDSAP

so ``systematics = SAP - F_cot`` is only the cotrending correction PDC applied.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import astropy.units as u
import lightkurve as lk
from astropy.utils.masked import Masked

__all__ = [
    "parse_kic",
    "compute_systematics",
    "systematics_from_lightcurve",
    "download_lightcurves",
    "get_systematics",
]

_EXPTIME = {"long": 1800, "short": 60}


def parse_kic(kic: int | str) -> int:
    """Return the integer KIC id from ``757450``, ``"KIC 757450"``, ``"kplr000757450"``, etc."""
    if isinstance(kic, (int, np.integer)):
        return int(kic)
    digits = re.findall(r"\d+", str(kic))
    if len(digits) != 1:
        raise ValueError(f"Could not parse a KIC id from {kic!r}")
    return int(digits[0])


def _as_float_array(x) -> np.ndarray:
    """Masked/Quantity/array -> plain float ndarray with masked entries set to NaN."""
    mask = None
    if isinstance(x, Masked):  # what lightkurve returns for columns with gaps
        x, mask = x.unmasked, np.asarray(x.mask)
    elif np.ma.isMaskedArray(x):
        x, mask = np.ma.getdata(x), np.ma.getmaskarray(x)
    arr = np.array(getattr(x, "value", x), dtype=float)
    if mask is not None:
        arr[mask] = np.nan
    return arr


def compute_systematics(
    sap_flux: np.ndarray,
    pdcsap_flux: np.ndarray,
    crowdsap: float = 1.0,
    flfrcsap: float = 1.0,
    undo_crowding: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Systematics for a single Kepler light curve file (one quarter or month).

    Parameters
    ----------
    sap_flux, pdcsap_flux : array
        Fluxes from the same file, in e-/s. NaNs are allowed and propagate.
    crowdsap, flfrcsap : float
        The file's ``CROWDSAP`` and ``FLFRCSAP`` header values.
    undo_crowding : bool
        If True (default), put PDCSAP back on the SAP scale before differencing
        (see module docstring). If False, return the literal ``SAP - PDCSAP``.

    Returns
    -------
    systematics : ndarray
        ``SAP - PDCSAP_rescaled`` in e-/s.
    pdcsap_rescaled : ndarray
        The PDCSAP flux on the SAP scale (equal to ``pdcsap_flux`` if
        ``undo_crowding`` is False).
    """
    sap = _as_float_array(sap_flux)
    pdc = _as_float_array(pdcsap_flux)
    if sap.shape != pdc.shape:
        raise ValueError(f"shape mismatch: sap {sap.shape} vs pdcsap {pdc.shape}")

    if undo_crowding:
        if not (0 < crowdsap <= 1 and 0 < flfrcsap <= 1):
            raise ValueError(
                f"CROWDSAP={crowdsap} and FLFRCSAP={flfrcsap} must be in (0, 1]"
            )
        # PDC subtracts (1-c)*median of the cotrended flux. On the PDCSAP scale
        # median(PDCSAP) = c * median(F_cot) / f, so the subtracted constant is
        # (1-c) * f * median(PDCSAP) / c.
        med_pdc = np.nanmedian(pdc) if np.any(np.isfinite(pdc)) else np.nan
        pdc = flfrcsap * pdc + (1.0 - crowdsap) * flfrcsap * med_pdc / crowdsap

    return sap - pdc, pdc


def systematics_from_lightcurve(
    lc: lk.LightCurve,
    normalize: bool = False,
    undo_crowding: bool = True,
    dropna: bool = True,
) -> lk.LightCurve:
    """Systematics for one Kepler light curve file loaded by lightkurve.

    ``lc`` must have ``sap_flux`` and ``pdcsap_flux`` columns and ``CROWDSAP``,
    ``FLFRCSAP`` and ``QUARTER`` in ``lc.meta``, as lightkurve's Kepler reader
    provides.

    Returns a ``LightCurve`` whose ``flux`` is the systematics. If ``normalize``
    is True, ``flux`` is divided by the median SAP flux of this file, which
    makes it a dimensionless fractional signal. The ``sap_flux``,
    ``pdcsap_flux`` and ``pdcsap_flux_rescaled`` columns always stay in e-/s.
    """
    meta = lc.meta
    crowdsap = float(meta.get("CROWDSAP", 1.0) or 1.0)
    flfrcsap = float(meta.get("FLFRCSAP", 1.0) or 1.0)
    quarter = int(meta.get("QUARTER", -1))

    sap = _as_float_array(lc["sap_flux"])
    sap_err = _as_float_array(lc["sap_flux_err"]) if "sap_flux_err" in lc.colnames else np.full_like(sap, np.nan)
    pdc = _as_float_array(lc["pdcsap_flux"])
    quality = np.asarray(_as_float_array(lc["quality"]), dtype=float) if "quality" in lc.colnames else np.zeros_like(sap)
    cadenceno = _as_float_array(lc["cadenceno"]) if "cadenceno" in lc.colnames else np.full_like(sap, -1)

    # Compute on finite cadences only, so the medians match what PDC saw.
    good = np.isfinite(sap) & np.isfinite(pdc)
    sys = np.full_like(sap, np.nan)
    pdc_rescaled = np.full_like(pdc, np.nan)
    sys[good], pdc_rescaled[good] = compute_systematics(
        sap[good], pdc[good], crowdsap, flfrcsap, undo_crowding=undo_crowding
    )

    if normalize:
        norm = np.nanmedian(sap[good]) if good.any() else np.nan
        flux = sys / norm
        flux_unit = u.dimensionless_unscaled
    else:
        flux = sys
        flux_unit = u.electron / u.s

    keep = good if dropna else np.ones_like(good)
    e_s = u.electron / u.s
    out = lk.LightCurve(
        time=lc.time[keep],
        flux=flux[keep] * flux_unit,
        flux_err=np.full(keep.sum(), np.nan) * flux_unit,
    )
    out["sap_flux"] = sap[keep] * e_s
    out["sap_flux_err"] = sap_err[keep] * e_s
    out["pdcsap_flux"] = pdc[keep] * e_s
    out["pdcsap_flux_rescaled"] = pdc_rescaled[keep] * e_s
    # Multiplicative systematics: SAP = ratio * (PDC-corrected flux on the SAP scale).
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ratio"] = (sap / pdc_rescaled)[keep] * u.dimensionless_unscaled
    out["quality"] = quality[keep].astype(np.int32)
    out["cadenceno"] = cadenceno[keep].astype(np.int64)
    out["quarter"] = np.full(keep.sum(), quarter, dtype=np.int16)

    out.meta.update(
        KEPLERID=meta.get("KEPLERID"),
        OBJECT=meta.get("OBJECT"),
        LABEL=meta.get("LABEL"),
        QUARTER=quarter,
        NORMALIZED=normalize,
        UNDO_CROWDING=undo_crowding,
    )
    out.meta.update(_header_values(meta))
    out.meta.update(CROWDSAP=crowdsap, FLFRCSAP=flfrcsap)
    return out


# Per-file FITS header values kept for CBV loading, donor matching and injection.
HEADER_KEYS = (
    "CROWDSAP", "FLFRCSAP", "MODULE", "OUTPUT", "CHANNEL", "SEASON",
    "KEPMAG", "PDCVAR", "CDPP3_0", "CDPP6_0", "CDPP12_0",
)


def _header_values(meta) -> dict:
    out = {}
    for key in HEADER_KEYS:
        val = meta.get(key)
        if isinstance(val, (np.integer, int)) and not isinstance(val, bool):
            val = int(val)
        elif isinstance(val, (np.floating, float)):
            val = float(val)
        out[key] = val
    return out


def download_lightcurves(
    kic: int | str,
    quarters: int | Iterable[int] | None = None,
    cadence: str = "long",
    quality_bitmask: str | int = "default",
    download_dir: str | None = None,
) -> lk.LightCurveCollection:
    """Download the official Kepler pipeline light curves for one KIC target.

    Files are cached by lightkurve (``~/.lightkurve/cache`` unless
    ``download_dir`` is given), so repeated calls do not hit MAST again.
    """
    if cadence not in _EXPTIME:
        raise ValueError(f"cadence must be one of {sorted(_EXPTIME)}, got {cadence!r}")
    kic_id = parse_kic(kic)
    if isinstance(quarters, (int, np.integer)):
        quarters = [int(quarters)]
    elif quarters is not None:
        quarters = [int(q) for q in quarters]

    search = lk.search_lightcurve(
        f"KIC {kic_id}",
        mission="Kepler",
        author="Kepler",
        exptime=_EXPTIME[cadence],
        quarter=quarters,
    )
    if len(search) == 0:
        q = "" if quarters is None else f" in quarter(s) {quarters}"
        raise LookupError(f"No Kepler {cadence}-cadence light curves found for KIC {kic_id}{q}")
    # search_lightcurve matches by position, so drop any nearby stars that came back.
    target = f"kplr{kic_id:09d}"
    search = search[np.asarray(search.target_name) == target]
    if len(search) == 0:
        raise LookupError(f"Search returned only nearby targets, none matching {target}")

    return search.download_all(quality_bitmask=quality_bitmask, download_dir=download_dir)


def get_systematics(
    kic: int | str,
    quarters: int | Sequence[int] | None = None,
    cadence: str = "long",
    normalize: bool = False,
    undo_crowding: bool = True,
    quality_bitmask: str | int = "default",
    dropna: bool = True,
    download_dir: str | None = None,
) -> lk.LightCurve:
    """Kepler instrumental systematics for a KIC target: SAP flux minus PDCSAP flux.

    Parameters
    ----------
    kic : int or str
        KIC id, e.g. ``11904151``, ``"KIC 11904151"`` or ``"kplr011904151"``.
    quarters : int or list of int, optional
        Quarters to include (0-17). Default: all available.
    cadence : {"long", "short"}
        30-min long cadence or 1-min short cadence.
    normalize : bool
        If False (default), return systematics in e-/s. If True, divide each
        quarter (or month, for short cadence) by its median SAP flux, giving a
        dimensionless fractional signal that can be added to a simulated
        light curve normalized to 1.
    undo_crowding : bool
        If True (default), undo PDC's crowding/flux-fraction correction before
        differencing, so the result holds only the cotrending correction. Set
        False for the literal ``SAP_FLUX - PDCSAP_FLUX``. See the module
        docstring.
    quality_bitmask : str or int
        Passed to lightkurve: "none", "default", "hard", "hardest" or an int.
    dropna : bool
        Drop cadences where SAP or PDCSAP is NaN.
    download_dir : str, optional
        lightkurve cache directory.

    Returns
    -------
    lightkurve.LightCurve
        ``flux`` holds the systematics (``flux_err`` is NaN; the difference
        has no independent pipeline uncertainty). Extra columns:
        ``sap_flux``, ``sap_flux_err``, ``pdcsap_flux``,
        ``pdcsap_flux_rescaled``, ``ratio`` (multiplicative systematics,
        ``sap_flux / pdcsap_flux_rescaled``), ``quality``, ``cadenceno``,
        ``quarter``. ``meta["CROWDSAP"]`` and ``meta["FLFRCSAP"]`` map each
        quarter to its header value; ``meta["QUARTER_META"]`` maps each
        quarter to all of ``HEADER_KEYS`` (CCD module/output, ``PDCVAR``, CDPP...).
    """
    kic_id = parse_kic(kic)
    lcs = download_lightcurves(
        kic_id,
        quarters=quarters,
        cadence=cadence,
        quality_bitmask=quality_bitmask,
        download_dir=download_dir,
    )
    pieces = [
        systematics_from_lightcurve(lc, normalize=normalize, undo_crowding=undo_crowding, dropna=dropna)
        for lc in lcs
    ]
    pieces.sort(key=lambda p: p.time.value[0] if len(p) else np.inf)

    # Collect per-file header values before stitching: with a single piece,
    # vstack can return that same object, so its meta must not be read afterwards.
    meta = dict(
        KEPLERID=kic_id,
        OBJECT=f"KIC {kic_id}",
        LABEL=f"KIC {kic_id}",
        CADENCE=cadence,
        NORMALIZED=normalize,
        UNDO_CROWDING=undo_crowding,
        QUALITY_BITMASK=quality_bitmask,
        QUARTERS=sorted({p.meta["QUARTER"] for p in pieces}),
        # Short cadence has three files per quarter; those values become lists.
        CROWDSAP=_per_quarter(pieces, "CROWDSAP"),
        FLFRCSAP=_per_quarter(pieces, "FLFRCSAP"),
        # All header values per quarter (first file of the quarter for short cadence).
        QUARTER_META=_quarter_meta(pieces),
    )
    out = lk.LightCurveCollection(pieces).stitch(corrector_func=lambda x: x)
    out.meta = meta
    return out


def _quarter_meta(pieces) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in pieces:
        out.setdefault(p.meta["QUARTER"], {k: p.meta.get(k) for k in HEADER_KEYS})
    return out


def _per_quarter(pieces, key):
    d: dict[int, list[float]] = {}
    for p in pieces:
        d.setdefault(p.meta["QUARTER"], []).append(p.meta[key])
    return {q: (v[0] if len(v) == 1 else v) for q, v in d.items()}
