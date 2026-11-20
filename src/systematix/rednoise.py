"""Red noise for S3, from the donors' residual spectra (design doc steps 2 and 4).

The step-2 fit (:func:`systematix.decompose.fit_flat`) leaves each donor a
residual ``r(t)`` in relative flux. Its periodogram, normalized so that white
noise of variance ``s**2`` has power ``s**2`` at every frequency (the
normalization of astropy's ``"psd"`` Lomb-Scargle with unit weights, which the
Phase 4 gate uses), is flat at high frequency, where only white noise is left
(``white``, the mean over WHITE_BAND), and rises above that floor at lower
frequency: granulation and low-level variability of the quiet star, and
systematics the model does not capture.

The residual is what is left after the fit, which removes most of the power at
the lowest frequencies (the polynomials, ramps and CBVs absorb it). Noise drawn
with the residual spectrum and fit again would lose that power twice, so S3
needs the noise before the fit. :func:`prefit_red` estimates it for one output
and quarter:

1. the power transfer ``T(f)`` of the fit, the mean periodogram of fit residuals
   over the spectrum of the input (white plus the binned red spectrum
   :func:`red_noise` is asked to synthesize), for noise with the output's
   typical spectrum on real donor designs (:func:`fit_transfer`), iterated
   because ``T`` depends on the input spectrum;
2. each donor's pre-fit excess over white, ``e(f) = P(f) / (T(f) white) - 1``;
3. the median excess shape ``R(f)`` (corrected for the skew of periodogram
   bins with few frequencies), and each donor's amplitude ``a``, a weighted
   least-squares fit of ``e = a R`` over all bins;
4. because stronger red noise is also steeper (on Q2 channel 65 the excess
   at 0.05 c/d runs from 6x white in the weakest amplitude quartile to 120x in
   the strongest), steps 1-3 are repeated within groups of donors of similar
   amplitude, each with its own shape and transfer.

A donor's red spectrum is then ``a R_g(f) white``: its own strength, the shape
of its amplitude group. Per-bin estimates from one quarter rest on one or two
Fourier frequencies at the lowest bins, so drawing a new realization from
them would compound two chi-square scatters and bias medians low.

``fit_flat``'s jitter is set by the whole residual variance, so it holds the
red variance too, spread over all frequencies as white noise. With red noise
on, S3 uses the white part of the jitter, :func:`white_jitter`
``= sqrt(max(white - mean(err**2), 0))``.

:func:`red_noise` draws a stationary Gaussian series with a binned spectrum on
one quarter's cadence grid: FFT synthesis over the full cadence-number range,
sampled at the cadences in use, so gaps do not change the spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaincinv

__all__ = ["LC_DT", "RED_EDGES", "WHITE_BAND", "binned_power", "bin_counts", "residual_spectra", "white_jitter",
           "cadence_grid", "red_noise", "fit_transfer", "prefit_red"]

LC_DT = 1765.4612 / 86400.0                     # Kepler long cadence (d)
RED_EDGES = np.geomspace(0.01, 24.4, 37)        # c/d; 1.24x wide, the gate's bin width, up to the LC Nyquist
WHITE_BAND = (8.0, 24.0)                        # c/d
T_MIN = 0.05                                    # bins the fit suppresses more than this do not set amplitudes


def _centres(edges):
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def _periodogram(R, cadenceno, dt):
    """(frequencies, power (n, nf), variance (n,), rows with data) of mean-subtracted, zero-filled rows."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    c = np.asarray(cadenceno, dtype=np.int64)
    c = c - c.min()
    N = int(c.max()) + 1
    ok = np.isfinite(R)
    n_used = ok.sum(axis=1)
    has = n_used > 0
    nu = np.where(has, n_used, 1)
    mu = np.where(ok, R, 0.0).sum(axis=1) / nu
    Y = np.where(ok, R - mu[:, None], 0.0)
    var = (Y ** 2).sum(axis=1) / nu
    X = np.zeros((len(R), N))
    X[:, c] = Y
    P = np.abs(np.fft.rfft(X, axis=1)) ** 2 / nu[:, None]
    return np.fft.rfftfreq(N, dt), P, var, has


def _bin_matrix(f, edges):
    edges = np.asarray(edges, dtype=float)
    nb = len(edges) - 1
    b = np.digitize(f, edges) - 1
    live = (f > 0) & (b >= 0) & (b < nb)
    B = np.zeros((live.sum(), nb))
    B[np.arange(live.sum()), b[live]] = 1.0
    return live, B


def _bin(f, P, edges):
    live, B = _bin_matrix(f, edges)
    cnt = B.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (P[:, live] @ B) / np.where(cnt > 0, cnt, np.nan)


def bin_counts(cadenceno, dt: float = LC_DT, edges=RED_EDGES) -> np.ndarray:
    """Number of Fourier frequencies of the cadence grid in each bin of ``edges``."""
    c = np.asarray(cadenceno, dtype=np.int64)
    f = np.fft.rfftfreq(int(c.max() - c.min()) + 1, dt)
    return _bin_matrix(f, edges)[1].sum(axis=0)


def binned_power(R, cadenceno, dt: float = LC_DT, edges=RED_EDGES):
    """Periodogram of the rows of ``R`` (n, m) averaged in bins of ``edges``: (n, nbins).

    NaN marks unused cadences. Each row is mean-subtracted, zero-filled onto
    the full cadence grid and Fourier transformed; power is ``|FFT|**2 / n_used``
    (white noise of variance s**2 gives s**2). Bins with no Fourier frequency
    are NaN, and so are rows with no used cadence.
    """
    f, P, _, has = _periodogram(R, cadenceno, dt)
    Pb = _bin(f, P, edges)
    Pb[~has] = np.nan
    return Pb


def residual_spectra(R, cadenceno, dt: float = LC_DT, edges=RED_EDGES, white_band=WHITE_BAND):
    """(white, red, var): white level (n,), post-fit red excess (n, nbins), variance (n,) of the rows of ``R``.

    ``red`` is the binned power minus ``white``, clipped at zero. It describes
    the residual as it is; for S3 use :func:`prefit_red`. Where a row has no red
    noise, sampling noise leaves a small positive excess: about 5% of the white
    level in bins with tens of Fourier frequencies, and up to ~37% in bins with one.
    """
    f, P, var, has = _periodogram(R, cadenceno, dt)
    band = (f >= white_band[0]) & (f <= white_band[1])
    if not band.any():
        raise ValueError(f"no frequency in the white band {white_band} c/d (Nyquist {f[-1]:.2f} c/d)")
    white = P[:, band].mean(axis=1)
    red = np.clip(_bin(f, P, edges) - white[:, None], 0.0, None)
    for a in (white, var):
        a[~has] = np.nan
    red[~has] = np.nan
    return white, red, var


def white_jitter(jitter, white, var):
    """White part of ``fit_flat``'s jitter: ``sqrt(max(white - mean(err**2), 0))``.

    ``fit_flat`` sets ``jitter**2 = var - mean(err**2)``, so
    ``mean(err**2) = var - jitter**2``.
    """
    jitter, white, var = (np.asarray(a, dtype=float) for a in (jitter, white, var))
    return np.sqrt(np.clip(white - (var - jitter ** 2), 0.0, None))


def cadence_grid(time, cadenceno=None):
    """(index of each cadence on a regular grid, grid length, cadence in days) for one quarter.

    Uses cadence numbers when they are valid (non-negative and increasing),
    otherwise the times.
    """
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64), 1, LC_DT
    c = None if cadenceno is None else np.asarray(cadenceno, dtype=np.int64)
    if c is not None and np.all(c >= 0) and np.all(np.diff(c) > 0) and t.size > 1:
        dt = float(np.median(np.diff(t) / np.diff(c)))
        idx = c - c.min()
    else:
        dt = float(np.median(np.diff(t))) if t.size > 1 else LC_DT
        idx = np.rint((t - t.min()) / dt).astype(np.int64)
    return idx, int(idx.max()) + 1, dt


def red_noise(red, idx, N: int, dt: float, rng, edges=RED_EDGES) -> np.ndarray:
    """A stationary Gaussian series with binned spectrum ``red``, at grid indices ``idx``.

    ``red`` is in the units of :func:`binned_power` (variance per Fourier
    frequency); it is interpolated between bin centres (log frequency) and held
    flat beyond the outer ones. The series has zero mean.
    """
    rng = np.random.default_rng(rng)
    red = np.asarray(red, dtype=float)
    fc = _centres(edges)
    ok = np.isfinite(red)
    if N < 2 or not ok.any() or not np.any(red[ok] > 0):
        return np.zeros(len(idx))
    f = np.fft.rfftfreq(N, dt)[1:]
    P = np.interp(np.log(f), np.log(fc[ok]), red[ok])
    a = np.sqrt(0.5 * N * P)
    X = np.zeros(len(f) + 1, dtype=complex)
    X[1:] = a * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    return np.fft.irfft(X, N)[np.asarray(idx, dtype=np.int64)]


def _intended(red, N: int, dt: float, edges=RED_EDGES) -> np.ndarray:
    """Binned spectrum that :func:`red_noise` is asked to synthesize on a grid of ``N`` cadences.

    The bin values of ``red`` where finite; elsewhere the bin average of
    red_noise's log-frequency interpolation; zeros if no bin is finite.
    Bins with no Fourier frequency on the grid are NaN.
    """
    edges = np.asarray(edges, dtype=float)
    red = np.asarray(red, dtype=float)
    fc = _centres(edges)
    ok = np.isfinite(red)
    f = np.fft.rfftfreq(N, dt)[1:]
    P = np.interp(np.log(f), np.log(fc[ok]), red[ok]) if ok.any() else np.zeros(len(f))
    b = np.digitize(f, edges) - 1
    live = (b >= 0) & (b < len(fc))
    cnt = np.bincount(b[live], minlength=len(fc)).astype(float)
    avg = np.bincount(b[live], weights=P[live], minlength=len(fc))
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(cnt > 0, avg / cnt, np.nan)
    return np.where(ok & (cnt > 0), red, avg)


def fit_transfer(designs, white: float, red, rng, n_draw: int = 8, edges=RED_EDGES) -> np.ndarray:
    """Power transfer of the Paper V fit per bin: mean periodogram of residuals over the input spectrum.

    ``designs`` is a list of ``(observation, err, D)``: a star's cadences, its
    relative flux errors and its design blocks (``decompose.design_blocks``).
    The input is white noise of variance ``white`` plus red noise with binned
    spectrum ``red``; the residual periodograms are taken on the star's own
    cadences.

    The denominator is the spectrum the input was drawn from, ``white + red``
    per bin (:func:`_intended`), not the input's periodogram on the star's
    cadences. Gaps, zero-filled on the cadence grid, leak the strong
    low-frequency power of a steep ``red`` into higher bins of the realized
    periodogram (by 1.05-1.7x at 0.08-0.7 c/d in the Phase 4 library), which
    made ``T`` too low there and :func:`prefit_red`'s red noise too strong by
    the same factor. Against the intended spectrum, ``T`` is exactly the factor
    that maps what :func:`red_noise` is given to what the fit leaves, which is
    what :func:`prefit_red` inverts.
    """
    from .decompose import fit_flat

    rng = np.random.default_rng(rng)
    nb = len(edges) - 1
    p_in, p_out = np.zeros(nb), np.zeros(nb)
    for obs, err, D in designs:
        idx, N, dt = cadence_grid(obs.time, obs.cadenceno)
        X = np.array([np.sqrt(white) * rng.standard_normal(len(idx)) + red_noise(red, idx, N, dt, rng, edges)
                      for _ in range(n_draw)])
        Res = np.array([fit_flat(x, err, D)[1] for x in X])
        Po = binned_power(Res, obs.cadenceno, dt, edges)
        Pi = white + _intended(red, N, dt, edges)
        live = np.isfinite(Po).all(axis=0) & np.isfinite(Pi)
        p_out[live] += Po[:, live].sum(axis=0)
        p_in[live] += n_draw * Pi[live]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p_in > 0, p_out / p_in, np.nan)


def _skew(counts):
    """Median of a bin's mean periodogram over its mean, for ``counts`` independent frequencies (Gamma)."""
    n = np.maximum(np.asarray(counts, dtype=float), 1.0)
    return gammaincinv(n, 0.5) / n


def _shape(rel, T, skew):
    """Median pre-fit excess over white per bin, corrected for the skew of few-frequency bins."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(np.nanmedian(rel / T, axis=0) / skew - 1.0, 0.0, None)


def _amplitude(rel, shape, T, counts):
    """Per-row weighted least-squares amplitude of the pre-fit excess ``rel / T - 1`` on ``shape`` (>= 0)."""
    fit = np.isfinite(shape) & (shape > 0) & np.isfinite(T) & (T >= T_MIN) & (counts > 0)
    if not fit.any():
        return np.zeros(len(rel))
    w = counts[fit] / (1.0 + shape[fit]) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        e = rel[:, fit] / T[fit] - 1.0
    e = np.where(np.isfinite(e), e, 0.0)
    return np.clip(e @ (w * shape[fit]) / np.sum(w * shape[fit] ** 2), 0.0, None)


def prefit_red(R, cadenceno, designs, rng, dt: float = LC_DT, edges=RED_EDGES, white_band=WHITE_BAND,
               iters: int = 3, n_draw: int = 8, n_groups: int = 8, min_group: int = 40):
    """Pre-fit red noise of residual rows ``R`` (n, m) from one output and quarter (module docstring).

    ``designs`` are ``(observation, err, D)`` triples for :func:`fit_transfer`.
    Donors are split into up to ``n_groups`` groups by red amplitude (at least
    ``min_group`` donors each), because stronger red noise is also steeper: each
    group gets its own shape and power transfer, and each donor's amplitude is
    refit against its group's shape.

    Returns a dict: ``white``, ``var``, ``amp`` and ``group`` (n,), ``red`` (n,
    nbins) = ``amp * shape[group] * white``, ``shape`` and ``transfer``
    (n_groups, nbins), and ``counts`` (nbins,).
    """
    rng = np.random.default_rng(rng)
    white, _, var = residual_spectra(R, cadenceno, dt, edges, white_band)
    Pb = binned_power(R, cadenceno, dt, edges)
    counts = bin_counts(cadenceno, dt, edges)
    skew = _skew(counts)
    use = np.isfinite(white) & (white > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = Pb / white[:, None]                                       # residual power in units of white
    w0 = float(np.nanmedian(white[use]))

    def converge(rows, T):
        for _ in range(iters):
            shape = _shape(rel[rows], T, skew)
            T = fit_transfer(designs, w0, shape * w0, rng, n_draw, edges)
        return _shape(rel[rows], T, skew), T

    shape, T = converge(use, np.ones(len(counts)))
    amp = _amplitude(rel, shape, T, counts)
    G = int(max(1, min(n_groups, use.sum() // min_group)))
    group = np.zeros(len(white), dtype=int)
    if G > 1:
        group = np.digitize(amp, np.quantile(amp[use], np.linspace(0, 1, G + 1)[1:-1]))
    shapes, transfers = np.zeros((G, len(counts))), np.zeros((G, len(counts)))
    for g in range(G):
        rows = group == g
        if G > 1:
            shape_g, T_g = converge(use & rows, T)
        else:
            shape_g, T_g = shape, T
        shapes[g], transfers[g] = shape_g, T_g
        amp[rows] = _amplitude(rel[rows], shape_g, T_g, counts)
    amp[~use] = 0.0
    red = amp[:, None] * np.nan_to_num(shapes[group]) * np.nan_to_num(white)[:, None]
    red[:, ~np.isfinite(transfers).all(axis=0)] = np.nan
    return {"white": white, "var": var, "amp": amp, "group": group, "red": red, "shape": shapes,
            "transfer": transfers, "counts": counts}
