"""Kepler instrumental systematics (SAP - PDCSAP) for any KIC target, plus the
instrument layer for injection-recovery tests: sources, detrenders and
transfer functions."""

import warnings as _warnings

# lightkurve warns on import about its optional PRF submodule; it does not affect us.
with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore", message=".*tpfmodel submodule.*")
    from .core import (
        HEADER_KEYS,
        compute_systematics,
        download_lightcurves,
        get_systematics,
        parse_kic,
        systematics_from_lightcurve,
    )
    from .observation import Observation
    from .sources import (
        DonorPool,
        NoSystematics,
        PDCRatio,
        QuietSAP,
        Systematics,
        inject,
        synthetic_cadence,
    )
    from .detrend import (
        ARC2,
        CBV,
        DETRENDERS,
        DetrendResult,
        Detrender,
        LkFlatten,
        Poly,
        QuarterMedian,
        Ramps,
        RollingMedian,
        make_detrender,
    )
    from .transfer import plot_transfer, signal_gain, transfer_sweep
    from .donors import load_nonperiodic, quiet_donors

__version__ = "0.2.0"

__all__ = [
    "HEADER_KEYS", "compute_systematics", "download_lightcurves", "get_systematics",
    "parse_kic", "systematics_from_lightcurve",
    "Observation",
    "Systematics", "inject", "synthetic_cadence", "NoSystematics", "PDCRatio", "QuietSAP", "DonorPool",
    "DetrendResult", "Detrender", "QuarterMedian", "RollingMedian", "LkFlatten", "Poly", "Ramps", "CBV", "ARC2",
    "DETRENDERS", "make_detrender",
    "signal_gain", "transfer_sweep", "plot_transfer",
    "load_nonperiodic", "quiet_donors",
]
