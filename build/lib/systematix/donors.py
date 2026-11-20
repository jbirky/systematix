"""Quiet donor stars: the non-periodic sample of McQuillan, Mazeh & Aigrain (2014).

McQuillan et al. (2014, ApJS 211, 24; arXiv:1402.5694) ran an autocorrelation
period search on 133,030 Kepler main-sequence targets. Their Table 2 lists
the 99,000 stars without a reliable period. 68,246 of those have no candidate
period at all (``Prot`` is NaN); the rest had a candidate that failed the
selection cuts, so they may still vary at a low level. The hare-and-hounds
exercise (Aigrain et al. 2015) drew its quiet stars from this sample.

``PDCVAR`` and CDPP from the light-curve headers are secondary cuts that match
a donor's noise level to the target's; they need the light curves, so they
are applied later (see :class:`systematix.sources.DonorPool`).
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ._cache import cache_dir, download

__all__ = ["NONPERIODIC_URL", "load_nonperiodic", "quiet_donors"]

NONPERIODIC_URL = "https://arxiv.org/src/1402.5694v2/anc/Table_2_Non_Periodic.txt"


def load_nonperiodic(path: str | None = None, cache: str | None = None) -> Table:
    """McQuillan et al. (2014) Table 2 as an astropy Table (downloaded once, then cached).

    Columns: ``KID, Teff, logg, Mass, Prot, Prot_err, LPH, w, DC``.
    """
    if path is None:
        path = download(NONPERIODIC_URL, cache_dir("donors", cache) / "mcquillan2014_table2.csv")
    return Table.read(path, format="ascii.csv")


def quiet_donors(
    table: Table | None = None,
    no_candidate: bool = True,
    teff: tuple[float | None, float | None] = (None, None),
    logg_min: float | None = None,
) -> np.ndarray:
    """KIC ids of quiet donor stars, sorted.

    Parameters
    ----------
    table : Table, optional
        Output of :func:`load_nonperiodic` (loaded if not given).
    no_candidate : bool
        Keep only stars with no candidate period at all (``Prot`` NaN).
    teff : (min, max)
        Effective temperature range in K; None leaves a side open.
    logg_min : float, optional
        Minimum surface gravity.
    """
    t = load_nonperiodic() if table is None else table
    keep = np.ones(len(t), dtype=bool)
    if no_candidate:
        prot = np.ma.filled(np.ma.asarray(t["Prot"], dtype=float), np.nan)
        keep &= ~np.isfinite(prot)
    teff_col = np.asarray(t["Teff"], dtype=float)
    if teff[0] is not None:
        keep &= teff_col >= teff[0]
    if teff[1] is not None:
        keep &= teff_col <= teff[1]
    if logg_min is not None:
        keep &= np.asarray(t["logg"], dtype=float) >= logg_min
    return np.sort(np.asarray(t["KID"], dtype=np.int64)[keep])
