"""Red noise from residual spectra (systematix.rednoise) and its place in Systematics."""
import numpy as np
import pytest

import systematix as sx
from systematix import rednoise as rn
from systematix.sources import bin_systematics

DT = rn.LC_DT
N = 4400
FC = np.sqrt(rn.RED_EDGES[:-1] * rn.RED_EDGES[1:])


def _gappy_cadences(seed=0):
    """A quarter of cadence numbers with two downlink gaps and scattered flagged cadences."""
    c = np.arange(N)
    keep = ~((c > 1400) & (c < 1460)) & ~((c > 2900) & (c < 2950))
    keep &= np.random.default_rng(seed).random(N) > 0.02
    return c[keep] + 5000


def _lorentz(amp=4e-7, f0=0.2):
    return amp / (1 + (FC / f0) ** 2)


def test_white_noise_has_a_flat_floor_and_no_red():
    cad = _gappy_cadences()
    sig = 3e-4
    R = np.random.default_rng(1).normal(0, sig, (50, len(cad)))
    white, red, var = rn.residual_spectra(R, cad)
    assert np.median(white) == pytest.approx(sig ** 2, rel=0.03)
    assert np.median(var) == pytest.approx(sig ** 2, rel=0.03)
    # only clipped noise is left (documented bias: ~5% of the floor where bins hold tens of frequencies)
    assert np.nanmean(red[:, FC > 0.5]) < 0.08 * sig ** 2
    assert np.nanmean(red[:, FC > 5.0]) < 0.03 * sig ** 2


def test_red_noise_reproduces_its_spectrum():
    cad = _gappy_cadences()
    idx, n, dt = rn.cadence_grid((cad - cad[0]) * DT, cad)
    assert n == cad[-1] - cad[0] + 1 and dt == pytest.approx(DT)
    target = _lorentz()
    rng = np.random.default_rng(2)
    X = np.array([rn.red_noise(target, idx, n, dt, rng) for _ in range(300)])
    assert abs(X.mean()) < 0.05 * X.std()
    _, red, _ = rn.residual_spectra(X, cad)
    band = (FC > 0.08) & (FC < 2.0)
    ratio = np.nanmean(red, axis=0)[band] / target[band]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
    assert np.all(np.abs(ratio - 1) < 0.2)


def test_white_jitter_is_the_white_part_of_the_jitter():
    cad = _gappy_cadences()
    idx, n, dt = rn.cadence_grid((cad - cad[0]) * DT, cad)
    rng = np.random.default_rng(3)
    err, jit_w = 2e-4, 1.5e-4
    red_t = _lorentz(amp=2e-6)
    R = np.array([rng.normal(0, np.hypot(err, jit_w), len(cad)) + rn.red_noise(red_t, idx, n, dt, rng)
                  for _ in range(40)])
    jitter = np.sqrt(np.clip(R.var(axis=1) - err ** 2, 0, None))        # what fit_flat reports
    white, red, var = rn.residual_spectra(R, cad)
    wj = rn.white_jitter(jitter, white, var)
    assert np.median(wj) == pytest.approx(jit_w, rel=0.05)
    assert np.all(wj < jitter)


def test_cadence_grid_falls_back_to_times():
    cad = _gappy_cadences()
    t = 100.0 + (cad - cad[0]) * DT
    a = rn.cadence_grid(t, cad)
    b = rn.cadence_grid(t, np.full(len(t), -1))
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == pytest.approx(b[2])


def test_bin_counts_match_the_fourier_grid():
    cad = _gappy_cadences()
    counts = rn.bin_counts(cad)
    f = np.fft.rfftfreq(cad[-1] - cad[0] + 1, DT)
    assert counts.sum() == np.sum((f > 0) & (f >= rn.RED_EDGES[0]) & (f < rn.RED_EDGES[-1]))
    assert counts[0] <= 2 and counts[-1] > 300


def test_flat_or_empty_spectra_give_no_noise():
    idx, n, dt = rn.cadence_grid(np.arange(100) * DT)
    rng = np.random.default_rng(0)
    assert np.all(rn.red_noise(np.zeros(len(FC)), idx, n, dt, rng) == 0)
    assert np.all(rn.red_noise(np.full(len(FC), np.nan), idx, n, dt, rng) == 0)


def _gappy_design(seed=0):
    """A one-quarter template on _gappy_cadences (gaps on the cadence grid) and its Paper V design blocks."""
    from systematix import decompose as dc

    cad = _gappy_cadences(seed)
    n = len(cad)
    meta = {"MODULE": 2, "OUTPUT": 1, "CHANNEL": 1, "CROWDSAP": 0.95, "KEPMAG": 13.0}
    tpl = sx.Observation(time=200.0 + (cad - cad[0]) * DT, flux=np.full(n, 1e4), flux_err=np.full(n, 1.0),
                         quarter=5, cadenceno=cad, quarter_meta={5: meta}, kic=1)
    cbv = {"cadenceno": cad, "vectors": np.random.default_rng(1).standard_normal((n, 16)) / np.sqrt(n),
           "gap": np.zeros(n, bool)}
    return tpl, dc.design_blocks(tpl, [], cbv_loader=lambda q, m, o: cbv), np.full(n, 1e-4)


def test_fit_transfer_handles_missing_red_bins():
    """NaN red bins (empty amplitude groups in short quarters) must not blank the transfer."""
    tpl, D, err = _gappy_design()
    live = rn.bin_counts(tpl.cadenceno) > 0
    for red in (np.where(FC < 0.05, np.nan, _lorentz()), np.full(len(FC), np.nan)):
        T = rn.fit_transfer([(tpl, err, D)], 4e-8, red, np.random.default_rng(0), n_draw=4)
        assert np.all(np.isfinite(T[live])) and np.all(np.isnan(T[~live]))


def test_prefit_red_round_trip_with_gaps_and_a_steep_spectrum():
    """Regression (2026-11-17): S3 refit must leave the donors' residual spectrum on a gappy cadence grid.

    With a steep red spectrum, gaps leak low-frequency power into higher bins of
    the realized input periodogram. A transfer measured against that periodogram
    was too low at 0.08-0.7 c/d, so the pre-fit red noise came out ~1.2-1.6x too
    strong (Phase 4 red_check, Q12). The transfer is now measured against the
    intended input spectrum. Over 10 seeds this ratio is 0.91-1.08 with the fix
    and 1.24-1.57 without.
    """
    from systematix import decompose as dc

    tpl, D, err = _gappy_design()
    n = len(tpl)
    idx, n_grid, dt = rn.cadence_grid(tpl.time, tpl.cadenceno)
    steep = 4e-8 * 40.0 * (FC / 0.1) ** -2.5           # 40x white at 0.1 c/d, ~10^4x at 0.01 c/d
    rng = np.random.default_rng(10)
    R = np.array([dc.fit_flat(rng.normal(0, 2e-4, n) + rn.red_noise(steep, idx, n_grid, dt, rng), err, D)[1]
                  for _ in range(200)])
    pr = rn.prefit_red(R, tpl.cadenceno, [(tpl, err, D)], rng, dt=dt)
    S = np.array([dc.fit_flat(np.sqrt(pr["white"][i]) * rng.standard_normal(n)
                              + rn.red_noise(pr["red"][i], idx, n_grid, dt, rng), err, D)[1]
                  for i in range(len(R)) for _ in range(2)])
    ratio = (np.nanmedian(rn.binned_power(S, tpl.cadenceno, dt), axis=0)
             / np.nanmedian(rn.binned_power(R, tpl.cadenceno, dt), axis=0))
    band = (FC > 0.1) & (FC < 1.0)
    assert np.exp(np.nanmean(np.log(ratio[band]))) == pytest.approx(1.0, abs=0.15)


def test_draw_noise_adds_red_and_binning_keeps_it():
    tpl = sx.synthetic_cadence(n_quarters=1, seed=0)
    n = len(tpl)
    sys = sx.Systematics(tpl, scale=np.ones(n), sigma=np.zeros(n), flux_err=np.full(n, 1e-4),
                         red=lambda rng: np.full(n, 5.0))
    np.testing.assert_allclose(sys.draw_noise(0), 5.0)
    b = bin_systematics(sys, 0.5)
    noise = b.draw_noise(0)
    assert len(noise) == len(b)
    np.testing.assert_allclose(noise, 5.0)
    plain = sx.Systematics(tpl, scale=np.ones(n), sigma=np.zeros(n), flux_err=np.full(n, 1e-4))
    assert plain.red is None and bin_systematics(plain, 0.5).red is None
