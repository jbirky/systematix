"""S3 generated source and its weight model (systematix.generated)."""
import numpy as np
import pytest

import systematix as sx
from systematix import decompose as dc
from systematix import rednoise as rn

CAD = 0.0204


def _template(q=5, t0=100.0, kic=1, seed=0, kepmag=13.0):
    rng = np.random.default_rng(seed)
    t = np.arange(t0, t0 + 30.0, CAD)
    t = t[(t < 115.0) | (t > 116.0)]
    n = len(t)
    meta = {"MODULE": 2, "OUTPUT": 1, "CHANNEL": 1, "CROWDSAP": 0.95}
    if kepmag is not None:
        meta["KEPMAG"] = kepmag
    return sx.Observation(time=t, flux=1e4 * (1 + 1e-4 * rng.standard_normal(n)), flux_err=np.full(n, 1.0),
                          quarter=q, cadenceno=np.arange(n) + 1000, quarter_meta={q: meta}, kic=kic)


def _cbv(n, seed=1):
    vec = np.random.default_rng(seed).standard_normal((n, 16)) / np.sqrt(n)
    return {"cadenceno": np.arange(n) + 1000, "vectors": vec, "gap": np.zeros(n, bool)}


def _red_spectrum(amp, f0=0.5):
    fc = np.sqrt(rn.RED_EDGES[:-1] * rn.RED_EDGES[1:])
    return amp / (1 + (fc / f0) ** 2)


def _decomposed_donors(n_donors=200, seed=2, steps=(108.0,), red_amp=0.0):
    """Donors whose true weights are Gaussian with known mean/sd, fit with fit_flat.

    Their noise is white (sd 2e-4 against reported errors of 1e-4), plus red
    noise with a Lorentzian spectrum of height ``red_amp`` when it is > 0.
    """
    rng = np.random.default_rng(seed)
    tpl = _template()
    cbv = _cbv(len(tpl))
    loader = lambda q, m, o: cbv                                          # noqa: E731
    D = dc.design_blocks(tpl, list(steps), cbv_loader=loader)
    M = np.hstack([D[b] for b in dc.BLOCKS])
    k = M.shape[1]
    true_mean = np.zeros(k)
    true_mean[-dc.N_CBV:] = 0.01
    true_sd = np.full(k, 1e-3)
    true_sd[-dc.N_CBV:] = 0.03
    t0c = dc.canonical_t0([[t for (_, t, _) in D["ramp_meta"]]])
    idx, n_grid, dt = rn.cadence_grid(tpl.time, tpl.cadenceno)
    spec = _red_spectrum(red_amp) if red_amp > 0 else None
    rows, R = [], []
    for i in range(n_donors):
        w = true_mean + true_sd * rng.standard_normal(k)
        rel = M @ w + rng.normal(0, 2e-4, len(tpl))
        if spec is not None:
            rel = rel + rn.red_noise(spec, idx, n_grid, dt, rng)
        fit, res, jit, _ = dc.fit_flat(rel, np.full(len(tpl), 1e-4), D)
        keys, kw, kC = dc.keyed_weights(fit, D, 5, t0c)
        rows.append((keys, kw, kC, jit))
        R.append(res)
    keys = rows[0][0]
    W = np.array([r[1] for r in rows])
    C = np.array([r[2] for r in rows])
    jitter = np.array([r[3] for r in rows])
    err = tpl.flux_err / np.median(tpl.flux)
    pr = rn.prefit_red(np.array(R), tpl.cadenceno, [(tpl, err, D)], np.random.default_rng(seed + 1), dt=dt)
    model = sx.WeightModel.estimate(5, 1, keys, np.arange(n_donors) + 100, W, C, jitter, list(steps), t0c,
                                    kepmag=np.linspace(12.0, 15.0, n_donors), red=pr["red"],
                                    white_jitter=rn.white_jitter(jitter, pr["white"], pr["var"]))
    return model, tpl, loader, D


@pytest.fixture(scope="module")
def donors():
    return _decomposed_donors()


@pytest.fixture(scope="module")
def red_donors():
    return _decomposed_donors(n_donors=120, seed=4, red_amp=2e-6)


def test_moment_model_recovers_spread(donors):
    model, *_ = donors
    cbv = model.kind == "cbv"
    assert np.median(model.sd[cbv]) == pytest.approx(0.03, rel=0.15)
    assert np.mean(model.mean[cbv]) == pytest.approx(0.01, abs=0.005)
    assert model.clipped < 0.05
    expect = np.sqrt(2e-4 ** 2 - 1e-4 ** 2)
    assert np.median(model.jitter) == pytest.approx(expect, rel=0.1)
    # white donors: the white part of the jitter is all of it
    assert model.has_red and np.median(model.white_jitter) == pytest.approx(expect, rel=0.1)


def test_realize_is_seeded(donors):
    model, tpl, loader, D = donors
    g = sx.Generated(tpl, {5: model}, design=D)
    a, b = g.realize(7), g.realize(7)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.draw_noise(3), b.draw_noise(3))
    assert not np.array_equal(a.scale, g.realize(8).scale)
    assert a.name == "S3:generated" and a.donor in model.kic and a.red is not None


def test_resample_uses_one_donors_weights(donors):
    model, tpl, loader, D = donors
    for red_noise, jit in ((True, model.white_jitter), (False, model.jitter)):
        g = sx.Generated(tpl, {5: model}, red_noise=red_noise, design=D)
        s, J, donor = g.draw_trend(np.random.default_rng(3))
        i = int(np.flatnonzero(model.kic == donor)[0])
        expect = g.X[5] @ model.W[i]
        expect -= np.median(expect)
        np.testing.assert_allclose(s, expect, atol=1e-12)
        assert J[5] == jit[i]


def test_all_blocks_off_reproduces_s0(donors):
    model, tpl, loader, D = donors
    g = sx.Generated(tpl, {5: model}, blocks=(), jitter=False, dilution=False, red_noise=False, design=D)
    s3 = g.realize(1)
    s0 = sx.NoSystematics(tpl, offsets=True).realize(1)
    np.testing.assert_allclose(s3.scale, s0.scale)
    np.testing.assert_allclose(s3.sigma, s0.sigma)
    np.testing.assert_allclose(s3.flux_err, s0.flux_err)
    assert s3.dilution == s0.dilution and s3.red is None


def test_single_block_and_gaussian(donors):
    model, tpl, loader, D = donors
    g = sx.Generated(tpl, {5: model}, blocks=("cbv",), design=D)
    assert np.all(g.X[5][:, model.kind != "cbv"] == 0)
    gg = sx.Generated(tpl, {5: model}, draw="gaussian", design=D)
    assert gg.kp_match is None
    sys = gg.realize(0)
    assert sys.donor is None and np.all(np.isfinite(sys.scale)) and np.all(np.isfinite(sys.draw_noise(0)))
    with pytest.raises(ValueError):
        sx.Generated(tpl, {5: model}, draw="bogus", design=D)


def test_save_load_and_subset(donors, tmp_path):
    model, tpl, loader, D = donors
    path = tmp_path / "Q05" / "ch01.h5"
    model.save(path)
    back = sx.load_weight_model(tmp_path, 5, 1)
    assert back.keys == model.keys
    np.testing.assert_allclose(back.W, model.W)
    np.testing.assert_allclose(back.cov, model.cov, rtol=1e-5, atol=1e-12)    # Ve stored as float32
    np.testing.assert_allclose(back.red, model.red, rtol=1e-6, atol=1e-20)     # red stored as float32
    np.testing.assert_allclose(back.white_jitter, model.white_jitter)
    np.testing.assert_allclose(back.red_edges, model.red_edges)
    half = model.subset(model.kic[::2])
    assert len(half.kic) == len(model.kic) // 2 and half.keys == model.keys
    assert half.red.shape == (len(half.kic), len(model.red_edges) - 1)
    # a model directory works in place of the dict (channel from quarter_meta)
    g = sx.Generated(tpl, tmp_path, cbv_loader=loader)
    assert np.all(np.isfinite(g.realize(0).scale))


def test_models_without_spectra_need_red_noise_off(donors):
    model, tpl, loader, D = donors
    bare = sx.WeightModel(model.quarter, model.channel, model.keys, model.kic, model.W, model.Ve, model.jitter,
                          model.steps, model.t0c, model.kepmag)
    with pytest.raises(ValueError, match="residual spectra"):
        sx.Generated(tpl, {5: bare}, design=D)
    g = sx.Generated(tpl, {5: bare}, red_noise=False, design=D)
    s, J, donor = g.draw_trend(np.random.default_rng(0))
    assert g.realize(0).red is None and J[5] == bare.jitter[np.flatnonzero(bare.kic == donor)[0]]


def test_kp_match_is_the_default_for_resampling(donors):
    model, tpl, loader, D = donors
    kp = np.linspace(10, 16, len(model.kic))
    m = sx.WeightModel(model.quarter, model.channel, model.keys, model.kic, model.W, model.Ve, model.jitter,
                       model.steps, model.t0c, kp, model.red, model.white_jitter, model.red_edges)
    t12 = _template(kepmag=12.0)
    g = sx.Generated(t12, {5: m}, design=D)
    assert g.kp_match == sx.KP_MATCH == 50
    rng = np.random.default_rng(0)
    used = {g.draw_weights(rng)[2] for _ in range(600)}
    near = set(m.kic[np.argsort(np.abs(kp - 12.0))[:50]].tolist())
    assert used <= near and len(used) > 30
    small = sx.Generated(t12, {5: m}, kp_match=20, design=D)
    assert {small.draw_weights(rng)[2] for _ in range(300)} <= set(m.kic[np.argsort(np.abs(kp - 12.0))[:20]].tolist())
    pooled = sx.Generated(t12, {5: m}, kp_match=None, design=D)
    assert pooled.kp_match is None and len({pooled.draw_weights(rng)[2] for _ in range(600)}) > 100
    with pytest.raises(ValueError):
        sx.Generated(t12, {5: m}, draw="gaussian", kp_match=20, design=D)
    with pytest.raises(ValueError, match="KEPMAG"):
        sx.Generated(_template(kepmag=None), {5: m}, design=D)
    assert np.allclose(m.subset(m.kic[:50]).kepmag, kp[:50])


def test_prefit_red_recovers_the_true_spectrum(red_donors):
    """The donors' red noise before the fit, from their residuals: transfer undone, white floor removed."""
    model, tpl, loader, D = red_donors
    assert np.median(model.white_jitter) == pytest.approx(np.sqrt(2e-4 ** 2 - 1e-4 ** 2), rel=0.1)
    assert np.median(model.white_jitter) < 0.9 * np.median(model.jitter)
    fc = np.sqrt(model.red_edges[:-1] * model.red_edges[1:])
    band = (fc > 0.1) & (fc < 3.0)          # bins with no Fourier frequency of the 30-d template are NaN
    ratio = np.nanmedian(model.red[:, band], axis=0) / _red_spectrum(2e-6)[band]
    assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.25)


def test_red_noise_matches_the_donors_residuals(red_donors):
    """S3 refit with the Paper V model leaves the donors' residual spectrum; without red noise it does not."""
    model, tpl, loader, D = red_donors
    idx, n_grid, dt = rn.cadence_grid(tpl.time, tpl.cadenceno)
    err = tpl.flux_err / np.median(tpl.flux)
    rng = np.random.default_rng(5)
    # the donors' own residuals, regenerated: true weights do not matter, only the noise
    true = [dc.fit_flat(rng.normal(0, 2e-4, len(tpl)) + rn.red_noise(_red_spectrum(2e-6), idx, n_grid, dt, rng),
                        err, D)[1] for _ in range(120)]
    P_true = np.nanmedian(rn.binned_power(np.array(true), tpl.cadenceno, dt), axis=0)
    true_var = 2e-4 ** 2 + np.mean([np.var(rn.red_noise(_red_spectrum(2e-6), idx, n_grid, dt, rng))
                                     for _ in range(200)])
    fc = np.sqrt(rn.RED_EDGES[:-1] * rn.RED_EDGES[1:])
    band = (fc > 0.1) & (fc < 3.0)
    got, var = {}, {}
    for red_noise in (True, False):
        g = sx.Generated(tpl, {5: model}, red_noise=red_noise, kp_match=None, design=D)
        R, V = [], []
        for _ in range(120):
            sys = g.realize(rng)
            noise = sys.draw_noise(rng)
            rel = (sys.scale + noise) / g.level - 1.0
            R.append(dc.fit_flat(rel, err, D)[1])
            V.append(np.var(noise / g.level))
        got[red_noise] = np.nanmedian(rn.binned_power(np.array(R), tpl.cadenceno, dt), axis=0)
        var[red_noise] = np.mean(V)
    assert np.nanmedian(got[True][band] / P_true[band]) == pytest.approx(1.0, abs=0.15)
    assert np.nanmedian(got[False][band] / P_true[band]) < 0.8
    # with red noise, S3's noise has the donors' variance before the fit; without, only what the fit left
    assert var[True] == pytest.approx(true_var, rel=0.15)
    assert var[False] < var[True]
