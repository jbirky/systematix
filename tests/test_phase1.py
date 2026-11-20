"""Instrument layer: Observation, sources, detrenders, transfer functions, CBV and donor loaders."""

import json
import warnings

import numpy as np
import pytest
from astropy.table import Table

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightkurve as lk  # noqa: F401

import systematix as sx
from systematix import cbv as cbvmod
from systematix import core, donors
from systematix.detrend import normalize_quarters, segment_poly_design

from test_core import fake_quarter


@pytest.fixture
def tmpl():
    return sx.synthetic_cadence(n_quarters=2, sigma=1e-4, seed=0)


@pytest.fixture
def sys0(tmpl):
    return sx.NoSystematics(tmpl).realize()


def sinusoid(t, P, A=1e-3, ph=0.3):
    return 1 + A * np.sin(2 * np.pi * t / P + ph)


# ---------------------------------------------------------------- Observation

def test_observation_validation_and_helpers(tmpl):
    assert tmpl.quarters == [1, 2]
    starts = tmpl.segment_starts(0.5)
    assert starts[0] == 0
    assert np.all(np.diff(tmpl.time)[starts[1:] - 1] > 0.5)
    assert np.all(tmpl.crowding() == 1)
    sub = tmpl.select(tmpl.quarter == 2)
    assert sub.quarters == [2]
    with pytest.raises(ValueError):
        sx.Observation(time=[0, 2, 1], flux=[1, 1, 1], flux_err=0.1, quarter=1)
    with pytest.raises(ValueError):
        sx.Observation(time=[0, 1, 2], flux=[1, 1], flux_err=0.1, quarter=1)


def test_observation_from_systematics_carries_header_meta(monkeypatch):
    lcs = []
    for q, (c, f) in {3: (0.9, 0.8), 4: (0.95, 0.85)}.items():
        lc, _, _ = fake_quarter(q, 100.0 * q, crowdsap=c, flfrcsap=f)
        lc.meta.update(MODULE=15, OUTPUT=4, CHANNEL=52, PDCVAR=0.5)
        lcs.append(lc)
    monkeypatch.setattr(core, "download_lightcurves", lambda *a, **k: lk.LightCurveCollection(lcs))
    sys_lc = sx.get_systematics(123)
    assert sys_lc.meta["QUARTER_META"][3]["MODULE"] == 15
    assert sys_lc.meta["QUARTER_META"][4]["CROWDSAP"] == 0.95
    # multiplicative ratio column
    np.testing.assert_allclose(sys_lc["ratio"].value,
                               sys_lc["sap_flux"].value / sys_lc["pdcsap_flux_rescaled"].value)
    obs = sx.Observation.from_systematics(sys_lc)
    assert obs.quarters == [3, 4]
    np.testing.assert_allclose(obs.crowding()[obs.quarter == 3], 0.9)
    np.testing.assert_allclose(obs.header("PDCVAR"), 0.5)


# ---------------------------------------------------------------- sources + injection

def test_inject_formula_and_dilution(tmpl):
    t = tmpl.replace(quarter_meta={1: {"CROWDSAP": 0.8}, 2: {"CROWDSAP": 0.9}})
    scale = np.where(t.quarter == 1, 1000.0, 2000.0) * (1 + 1e-3 * t.time / 90)
    sys = sx.Systematics(t, scale=scale, sigma=np.zeros(len(t)), flux_err=np.ones(len(t)))
    f = sinusoid(t.time, 7.0)
    obs = sx.inject(sys, f)
    c = np.where(t.quarter == 1, 0.8, 0.9)
    np.testing.assert_allclose(obs.flux, scale * (c * f + 1 - c))
    nodil = sx.inject(sx.Systematics(t, scale, np.zeros(len(t)), np.ones(len(t)), dilution=False), f)
    np.testing.assert_allclose(nodil.flux, scale * f)
    assert obs.truth["f_spot"] is not f and np.allclose(obs.truth["f_spot"], f)


def test_paired_noise(sys0):
    noise = sys0.draw_noise(5)
    a = sx.inject(sys0, 1.0, noise)
    b = sx.inject(sys0, 1.0, noise)
    np.testing.assert_array_equal(a.flux, b.flux)
    assert np.std(noise) == pytest.approx(1e-4, rel=0.05)


def _fake_systematics_lc():
    lcs = []
    for q in (2, 3):
        lc, sys, star = fake_quarter(q, 100.0 * q, crowdsap=0.9, flfrcsap=0.85, seed=q)
        lc.meta.update(MODULE=2, OUTPUT=1)
        lcs.append(lc)
    pieces = [core.systematics_from_lightcurve(lc) for lc in lcs]
    out = lk.LightCurveCollection(pieces).stitch(corrector_func=lambda x: x)
    out.meta = {"KEPLERID": 123, "QUARTER_META": core._quarter_meta(pieces)}
    return out


def test_pdc_ratio_source():
    lc = _fake_systematics_lc()
    sys = sx.PDCRatio(lc).realize()
    assert sys.name.startswith("S1")
    # Without a signal or noise, the injected flux reproduces SAP up to the donor's own star.
    obs = sx.inject(sys, 1.0, noise=np.zeros(len(sys)))
    ratio = lc["ratio"].value
    for q in (2, 3):
        m = obs.quarter == q
        expect = np.median(lc["pdcsap_flux_rescaled"].value[lc["quarter"] == q]) * ratio[lc["quarter"] == q]
        np.testing.assert_allclose(obs.flux[m], expect)
    np.testing.assert_allclose(sys.sigma, lc["sap_flux_err"].value)


def test_quiet_sap_source():
    lc = _fake_systematics_lc()
    sys = sx.QuietSAP(lc).realize()
    obs = sx.inject(sys, 1.0)
    np.testing.assert_allclose(obs.flux, lc["sap_flux"].value)   # no noise added
    obs2 = sx.inject(sys, sinusoid(sys.time, 5.0, A=1e-2))
    rel = obs2.flux / obs.flux - 1
    np.testing.assert_allclose(rel, 0.9 * 1e-2 * np.sin(2 * np.pi * sys.time / 5.0 + 0.3), atol=1e-12)


def test_donor_pool_rejects_noisy_donors(monkeypatch):
    lc_quiet = _fake_systematics_lc()
    lc_noisy = _fake_systematics_lc()
    for qm in lc_quiet.meta["QUARTER_META"].values():
        qm["PDCVAR"] = 0.1
    for qm in lc_noisy.meta["QUARTER_META"].values():
        qm["PDCVAR"] = 10.0
    lc_noisy.meta["KEPLERID"] = 456
    lookup = {123: lc_quiet, 456: lc_noisy}
    monkeypatch.setattr(sx.sources, "_load_donor", lambda kic, q, d: lookup[int(kic)])
    pool = sx.DonorPool([123, 456], kind="quiet_sap", max_pdcvar=1.0, max_tries=50)
    for seed in range(5):
        assert pool.realize(seed).donor == 123


# ---------------------------------------------------------------- detrenders

def test_quarter_median_removes_offsets(tmpl):
    obs = tmpl.replace(flux=np.where(tmpl.quarter == 1, 1000.0, 3000.0) * sinusoid(tmpl.time, 3.0))
    r = sx.QuarterMedian()(obs)
    for q in (1, 2):
        assert np.median(r.flux[r.quarter == q]) == pytest.approx(1.0, abs=1e-6)


def test_poly_subtract_and_marginalize(tmpl):
    x = tmpl.time
    trend = np.where(tmpl.quarter == 1, 1e-3 * ((x - 45) / 45) ** 2, -2e-3 * (x - 136) / 45)
    obs = tmpl.replace(flux=1000.0 * (1 + trend))
    r = sx.Poly(2)(obs)
    np.testing.assert_allclose(r.flux, 1.0, atol=1e-9)
    m = sx.Poly(2)(obs, mode="marginalize")
    assert m.design.shape == (len(obs), 6)
    assert np.all(np.isinf(m.prior_sigma))
    rel, _ = normalize_quarters(obs)
    np.testing.assert_allclose(m.flux, 1 + rel)


def test_segment_poly_design_matches_spotgp_convention():
    t = np.linspace(0, 10, 11)
    lab = np.r_[np.zeros(5), np.ones(6)]
    M = segment_poly_design(t, lab, order=1)
    assert M.shape == (11, 4)
    assert np.allclose(np.abs(M[:5, 1]).max(), 1) and np.allclose(M[5:, :2], 0)


def test_ramps_remove_thermal_transients(tmpl):
    starts = tmpl.segment_starts(0.5)
    ends = np.r_[starts[1:], len(tmpl)]
    ramp = np.zeros(len(tmpl))
    for s, e in zip(starts, ends):
        ramp[s:e] = -2e-3 * np.exp(-(tmpl.time[s:e] - tmpl.time[s]) / 2.0)
    obs = tmpl.replace(flux=500.0 * (1 + ramp))
    r = sx.Ramps(taus=(2.0,), order=0)(obs)
    np.testing.assert_allclose(r.flux, 1.0, atol=1e-9)


def _fake_cbv_loader(tmpl, n=16, seed=0):
    rng = np.random.default_rng(seed)
    cad = tmpl.cadenceno
    full = np.arange(cad.min(), cad.max() + 1)
    vecs = {}
    for q in tmpl.quarters:
        V = rng.standard_normal((full.size, n)).cumsum(axis=0)
        V -= V.mean(axis=0)
        V /= np.linalg.norm(V, axis=0)
        vecs[q] = V

    def loader(q, module, output):
        assert (module, output) == (2, 1)
        return {"cadenceno": full, "vectors": vecs[q], "gap": np.zeros(full.size, bool)}
    return loader, full, vecs


def test_cbv_design_alignment_and_fit(tmpl):
    t = tmpl.replace(quarter_meta={q: {"MODULE": 2, "OUTPUT": 1} for q in tmpl.quarters})
    loader, full, vecs = _fake_cbv_loader(t)
    M = cbvmod.cbv_design(t, n_cbv=4, loader=loader)
    assert M.shape == (len(t), 8)
    idx = np.searchsorted(full, t.cadenceno)
    q1 = t.quarter == 1
    np.testing.assert_allclose(M[q1, :4], vecs[1][idx[q1], :4])
    assert np.all(M[q1, 4:] == 0)
    # A cadence missing from the CBV table gets a zero row.
    t_missing = t.replace(cadenceno=np.where(np.arange(len(t)) == 10, 10**9, t.cadenceno))
    assert np.all(cbvmod.cbv_design(t_missing, 4, loader=loader)[10] == 0)

    sysm = M @ np.array([0.05, -0.03, 0.02, 0.01, -0.04, 0.02, 0.0, 0.01])
    obs = t.replace(flux=800.0 * (1 + sysm), flux_err=np.full(len(t), 800.0 * 1e-4))
    flat = sx.CBV(n_cbv=4, loader=loader)(obs)
    np.testing.assert_allclose(flat.flux, 1.0, atol=1e-6)
    tight = sx.CBV(n_cbv=4, prior_sigma=1e-6, loader=loader)(obs)
    assert np.std(tight.flux) > 0.5 * np.std(sysm)        # strong prior leaves it in
    marg = sx.CBV(n_cbv=4, prior_sigma=0.1, loader=loader)(obs, mode="marginalize")
    assert marg.design.shape[1] == 2 + 8
    assert np.all(np.isinf(marg.prior_sigma[:2])) and np.all(marg.prior_sigma[2:] == 0.1)


def test_rolling_median_and_flatten_are_high_pass(tmpl, sys0):
    noise = np.zeros(len(sys0))
    for d in (sx.RollingMedian(10.0), sx.LkFlatten(2.0)):
        fast, _ = sx.signal_gain(d, sys0, sinusoid(sys0.time, 0.5), noise)
        slow, _ = sx.signal_gain(d, sys0, sinusoid(sys0.time, 60.0), noise)
        assert fast > 0.8 and slow < 0.1, d.label


def test_clip_drops_outliers(tmpl):
    flux = np.ones(len(tmpl)) + np.random.default_rng(0).normal(0, 1e-4, len(tmpl))
    flux[[100, 2000]] = 1.01
    r = sx.QuarterMedian(clip=5)(tmpl.replace(flux=flux))
    assert len(r.time) == len(tmpl) - 2 and not r.mask[100]


def test_make_detrender_and_modes(tmpl):
    d = sx.make_detrender({"name": "poly", "order": 1, "mode": "marginalize"})
    assert isinstance(d, sx.Poly) and d.order == 1 and d.label == "poly(order=1)"
    with pytest.raises(KeyError):
        sx.make_detrender("nope")
    with pytest.raises(ValueError):
        sx.RollingMedian()(tmpl, mode="marginalize")
    with pytest.raises(ValueError):
        sx.Poly()(tmpl, mode="other")


# ---------------------------------------------------------------- transfer functions

def test_transfer_quarter_median_keeps_short_loses_long(sys0):
    tab = sx.transfer_sweep([sx.QuarterMedian(), sx.Poly(2)], sys0, periods=[2.0, 300.0], n_phase=3)
    qm = tab[tab["detrender"] == "quarter_median"]
    assert qm["gain"][0] == pytest.approx(1.0, abs=0.01)
    assert qm["power"][1] < 0.3
    pl = tab[tab["detrender"] == "poly(order=2)"]
    assert pl["power"][1] < qm["power"][1]
    assert set(tab.colnames) == {"detrender", "period", "gain", "gain_std", "power", "distortion"}


def test_transfer_linear_detrender_independent_of_systematics(tmpl):
    """For a linear detrender the paired difference cancels the systematics exactly."""
    t = tmpl
    scale = 1000.0 * (1 + 3e-3 * np.sin(2 * np.pi * t.time / 40.0))
    sys_a = sx.Systematics(t, np.full(len(t), 1000.0), np.zeros(len(t)), np.full(len(t), 0.1))
    sys_b = sx.Systematics(t, scale, np.zeros(len(t)), np.full(len(t), 0.1))
    f = sinusoid(t.time, 9.0)
    ga, _ = sx.signal_gain(sx.Poly(3), sys_a, f, np.zeros(len(t)))
    gb, _ = sx.signal_gain(sx.Poly(3), sys_b, f, np.zeros(len(t)))
    assert ga == pytest.approx(gb, abs=5e-3)


# ---------------------------------------------------------------- loaders (offline)

def test_quiet_donors_filters(tmp_path):
    p = tmp_path / "t2.csv"
    p.write_text("KID,Teff,logg,Mass,Prot,Prot_err,LPH,w,DC\n"
                 "1,5800,4.4,1.0,nan,nan,nan,nan,0\n"
                 "2,5800,4.4,1.0,12.3,0.5,0.2,0.1,0\n"
                 "3,4000,4.6,0.6,nan,nan,nan,nan,0\n"
                 "4,6200,3.9,1.2,nan,nan,nan,nan,0\n")
    t = donors.load_nonperiodic(path=str(p))
    assert list(sx.quiet_donors(t)) == [1, 3, 4]
    assert list(sx.quiet_donors(t, no_candidate=False)) == [1, 2, 3, 4]
    assert list(sx.quiet_donors(t, teff=(5000, 6000), logg_min=4.2)) == [1]


def test_cbv_filename_from_cached_listing(tmp_path):
    d = tmp_path / "cbv"
    d.mkdir()
    (d / "listing.json").write_text(json.dumps(
        ["kplr2009350155506-q03-d25_lcbv.fits", "kplr2010078095331-q04-d25_lcbv.fits"]))
    assert cbvmod.cbv_filename(3, cache=str(tmp_path)) == "kplr2009350155506-q03-d25_lcbv.fits"
    with pytest.raises(LookupError):
        cbvmod.cbv_filename(5, cache=str(tmp_path))


# ---------------------------------------------------------------- live MAST / arXiv

@pytest.mark.remote
def test_live_donor_table(tmp_path):
    t = donors.load_nonperiodic(cache=str(tmp_path))
    assert len(t) == 99000
    assert len(sx.quiet_donors(t)) == 68246


@pytest.mark.remote
def test_live_cbv_fit_kepler10(tmp_path):
    lc = sx.get_systematics(11904151, quarters=[3], download_dir=str(tmp_path))
    obs = sx.Observation.from_systematics(lc)
    assert obs.quarter_meta[3]["MODULE"] == 15 and obs.quarter_meta[3]["OUTPUT"] == 4
    M = cbvmod.cbv_design(obs, n_cbv=8, loader=lambda q, m, o: cbvmod.load_cbv_quarter(q, m, o, str(tmp_path)))
    assert M.shape == (len(obs), 8) and np.all(np.any(M != 0, axis=0))
    # The SAP flux is mostly explained by the CBVs plus an offset.
    r = sx.CBV(8, cache=str(tmp_path))(obs)
    assert np.std(r.flux) < 0.5 * np.std(sx.QuarterMedian()(obs).flux)


# ---------------------------------------------------------------- binning and clip methods

def test_bin_systematics_matches_inject_then_bin(sys0):
    f = sinusoid(sys0.time, 7.0, A=1e-2)
    b = sx.sources.bin_systematics(sys0, 0.2)
    assert len(b) < len(sys0) / 5
    assert set(np.unique(b.template.quarter)) == set(sys0.template.quarters)
    # Noise averages down as 1/sqrt(n) per bin.
    n_per = len(sys0) / len(b)
    assert np.median(b.sigma) == pytest.approx(1e-4 / np.sqrt(n_per), rel=0.2)
    # Injecting at bin centres into the binned scale = binning the injected scale (scale = 1 here).
    obs_b = sx.inject(b, sinusoid(b.time, 7.0, A=1e-2), noise=np.zeros(len(b)))
    ids, w, nb = sx.sources.bin_weights(sys0.time, sys0.template.quarter, sys0.flux_err, 0.2)
    fine = np.bincount(ids, weights=w * f, minlength=nb) / np.bincount(ids, weights=w, minlength=nb)
    np.testing.assert_allclose(obs_b.flux, fine, atol=2e-5)


def test_binned_result_keeps_linear_model_exact(tmpl):
    x = tmpl.time
    trend = np.where(tmpl.quarter == 1, 1e-3 * ((x - 45) / 45) ** 2, -2e-3 * (x - 136) / 45)
    obs = tmpl.replace(flux=1000.0 * (1 + trend))
    res = sx.Poly(2)(obs, mode="marginalize")
    b = res.binned(0.5)
    assert b.design.shape == (len(b.time), 6)
    # A linear combination of design columns bins to the same combination of binned columns.
    w = np.linalg.lstsq(res.design, res.flux - 1, rcond=None)[0]
    np.testing.assert_allclose(b.design @ w, b.flux - 1, atol=1e-9)


def test_std_clip_matches_one_pass_mean_std(tmpl):
    rng = np.random.default_rng(0)
    flux = 1 + rng.normal(0, 1e-4, len(tmpl))
    flux[[10, 20]] = 1.002
    obs = tmpl.replace(flux=flux)
    r = sx.QuarterMedian(clip=3, clip_method="std")(obs)
    rel, _ = sx.detrend.normalize_quarters(obs)
    m, s = rel.mean(), rel.std()
    expect = (rel >= m - 3 * s) & (rel <= m + 3 * s)
    np.testing.assert_array_equal(r.mask, expect)
    assert sx.make_detrender({"name": "rolling_median", "clip": 3, "clip_method": "std"}).label \
        == "rolling_median(window=10, clip=3, clip_method=std)"
    with pytest.raises(ValueError):
        sx.QuarterMedian(clip_method="sigma")
