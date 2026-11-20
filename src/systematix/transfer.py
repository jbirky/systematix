"""How much of an injected signal survives each detrender (no fitting needed).

For a signal ``f(t)``, each trial runs the detrender twice on the same
systematics and the same noise, with and without the signal. The difference
of the two outputs is the *recovered* signal, so systematics and noise cancel
(exactly for linear detrenders, approximately for nonlinear ones). It is
compared with the *injected* signal as it appears in relative flux,
``c_q (f - 1) scale / median_q(scale)``:

* ``gain = <rec, inj> / <inj, inj>``: in-phase amplitude retained (1 = kept);
  power retained is ``gain**2``.
* ``distortion = 1 - <rec, inj>^2 / (<rec, rec> <inj, inj>)``: share of the
  recovered signal that is not a scaled copy of the injected one.

Per-quarter normalization counts as detrending here: a signal much longer
than a quarter is mostly absorbed into the quarter offsets.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from .detrend import Detrender, make_detrender
from .sources import Systematics, inject

__all__ = ["signal_gain", "sinusoid_signals", "transfer_sweep", "plot_transfer"]


def _injected_relative(sys: Systematics, f: np.ndarray) -> np.ndarray:
    t = sys.template
    inj = np.empty(len(t))
    c = sys.crowding()
    for q, m in t.quarter_masks().items():
        inj[m] = c[m] * (f[m] - 1.0) * sys.scale[m] / np.nanmedian(sys.scale[m])
    return inj


def signal_gain(detrender: Detrender, sys: Systematics, f: np.ndarray, noise: np.ndarray,
                baseline=None) -> tuple[float, float]:
    """(gain, distortion) for one signal ``f`` (normalized to 1) under ``sys``.

    ``baseline`` is the detrender's output without the signal (computed if
    not given); pass it to reuse it across signals.
    """
    if baseline is None:
        baseline = detrender(inject(sys, 1.0, noise))
    out = detrender(inject(sys, f, noise))
    rec = out.full() - baseline.full()
    inj = _injected_relative(sys, np.asarray(f, dtype=float))
    ok = np.isfinite(rec) & np.isfinite(inj)
    rec, inj = rec[ok], inj[ok]
    ii, ri, rr = inj @ inj, rec @ inj, rec @ rec
    if ii == 0:
        return np.nan, np.nan
    gain = ri / ii
    distortion = 1.0 - ri * ri / (rr * ii) if rr > 0 else 1.0
    return float(gain), float(distortion)


def sinusoid_signals(time: np.ndarray, period: float, amplitude: float = 1e-3, n_phase: int = 4, rng=None):
    """``n_phase`` sinusoids ``1 + A sin(2 pi t / P + phi)`` with evenly spaced, randomly offset phases."""
    rng = np.random.default_rng(rng)
    phases = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(n_phase) / n_phase
    return [1.0 + amplitude * np.sin(2 * np.pi * time / period + ph) for ph in phases]


def transfer_sweep(detrenders, sys: Systematics, periods, amplitude: float = 1e-3,
                   n_phase: int = 4, seed: int = 0) -> Table:
    """Sinusoid gain versus period for each detrender, on paired trials.

    ``detrenders`` is a list of :class:`Detrender` objects or config dicts.
    Every detrender sees the same noise and the same signals.

    Returns a Table with columns ``detrender, period, gain, gain_std,
    power, distortion``.
    """
    rng = np.random.default_rng(seed)
    noise = sys.draw_noise(rng)
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    signals = {P: sinusoid_signals(sys.time, P, amplitude, n_phase, rng) for P in periods}
    rows = []
    for d in detrenders:
        d = d if isinstance(d, Detrender) else make_detrender(d)
        base = d(inject(sys, 1.0, noise))
        for P in periods:
            res = np.array([signal_gain(d, sys, f, noise, base) for f in signals[P]])
            g = res[:, 0]
            rows.append((d.label, P, np.nanmean(g), np.nanstd(g), np.nanmean(g ** 2), np.nanmean(res[:, 1])))
    return Table(rows=rows, names=("detrender", "period", "gain", "gain_std", "power", "distortion"))


def plot_transfer(table: Table, ax=None, title: str | None = None):
    """Power retained versus period, one line per detrender."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for name in dict.fromkeys(table["detrender"]):
        t = table[table["detrender"] == name]
        ax.plot(t["period"], np.clip(t["power"], 1e-4, None), marker="o", ms=3, lw=1.5, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("Injected period [d]")
    ax.set_ylabel("Power retained (gain$^2$)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    return ax
