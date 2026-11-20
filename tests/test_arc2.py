"""ARC2 CBV correction: native reimplementation, detrender, optional OxKeplerSC backend.

The cross-checks against OxKeplerSC run only when it is available (installed,
or cloned at $OXKEPLERSC_PATH); otherwise they are skipped.
"""

import os
import sys
import warnings

import numpy as np
import pytest

import systematix as sx
from systematix import _oxksc
from systematix.arc2 import fit_cbv_weights, medransig, select_n_cbv, vb_linear_ard

needs_oxksc = pytest.mark.skipif(not _oxksc.available(),
                                 reason="OxKeplerSC not available (set OXKEPLERSC_PATH)")


def smooth_basis(n, k, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, k)).cumsum(axis=0)
    V -= V.mean(axis=0)
    return V / np.linalg.norm(V, axis=0)


# ---------------------------------------------------------------- core numerics

def test_vb_ard_recovers_relevant_and_shrinks_irrelevant():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 6))
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    y = X @ w_true + rng.normal(0, 0.1, 2000)
    fit = vb_linear_ard(X, y)
    np.testing.assert_allclose(fit["w"][:3], w_true[:3], atol=0.01)
    assert np.all(np.abs(fit["w"][3:]) < 0.01)
    # irrelevant weights get much larger prior precision
    assert fit["E_alpha"][3:].min() > 100 * fit["E_alpha"][:3].max()
    assert fit["E_tau"] == pytest.approx(1 / 0.1 ** 2, rel=0.1)
    assert fit["n_iter"] < 500


def test_fit_cbv_weights_is_scale_and_offset_invariant():
    B = smooth_basis(1500, 4)
    flux = 5e4 + 1e4 * (B @ np.array([0.3, -0.2, 0.1, 0.05])) + np.random.default_rng(0).normal(0, 5, 1500)
    w1 = fit_cbv_weights(flux, B)
    w2 = fit_cbv_weights(3.0 * flux + 7.0, B)
    np.testing.assert_allclose(w2, 3.0 * w1, rtol=1e-8)


def test_medransig():
    x = 1000.0 * (1 + np.r_[np.linspace(-1e-3, 1e-3, 1001)])
    med, ran, sig = medransig(x)
    assert med == pytest.approx(1000.0)
    assert ran == pytest.approx(1800.0, rel=0.01)       # 5-95% of a 2000 ppm ramp
    assert sig == pytest.approx(1.48 * 2.0, rel=0.01)   # constant 2 ppm steps
    med2, ran2, _ = medransig(np.r_[x, np.nan, np.nan])
    assert (med2, ran2) == (med, ran)


def test_select_n_cbv_picks_enough_vectors():
    rng = np.random.default_rng(3)
    B = smooth_basis(3000, 8, seed=4)
    trend = B[:, :3] @ np.array([4e3, -2.5e3, 1.5e3])
    flux = 2e5 + trend + rng.normal(0, 20, 3000)
    fit = select_n_cbv(flux, B, n_max=8)
    assert 3 <= fit.n_cbv <= 8
    resid = fit.corrected - np.median(fit.corrected)
    assert np.std(resid) < 1.5 * 20
    assert fit.corrected_all.shape == (8, 3000) and fit.weights.shape == (fit.n_cbv,)
    assert np.median(fit.corrected) == pytest.approx(np.median(flux), rel=1e-6)


# ---------------------------------------------------------------- detrender

def _obs_with_cbv_trend(seed=0):
    tmpl = sx.synthetic_cadence(n_quarters=2, sigma=1e-4, seed=seed)
    tmpl = tmpl.replace(quarter_meta={q: {"MODULE": 2, "OUTPUT": 1} for q in tmpl.quarters})
    full = np.arange(tmpl.cadenceno.min(), tmpl.cadenceno.max() + 1)
    vecs = {q: smooth_basis(full.size, 16, seed=10 + q) for q in tmpl.quarters}

    def loader(q, module, output):
        return {"cadenceno": full, "vectors": vecs[q], "gap": np.zeros(full.size, bool)}

    V = sx.detrend.CBV(n_cbv=8, loader=loader)._cbv_matrix(tmpl, 8)
    w = np.r_[[0.4, -0.3, 0.2, 0, 0, 0, 0, 0], [-0.2, 0.25, 0, 0.1, 0, 0, 0, 0]]
    rel_trend = V @ w
    noise = np.random.default_rng(seed).normal(0, 1e-4, len(tmpl))
    obs = tmpl.replace(flux=1e5 * (1 + rel_trend + noise), flux_err=np.full(len(tmpl), 10.0))
    return obs, loader, rel_trend


def test_arc2_detrender_removes_cbv_trend():
    obs, loader, rel_trend = _obs_with_cbv_trend()
    d = sx.ARC2(n_max=8, loader=loader)
    res = d(obs)
    assert np.std(rel_trend) > 10 * 1e-4
    assert np.std(res.flux) < 1.3e-4
    assert set(res.meta["n_cbv_selected"]) == {1, 2}
    assert all(1 <= n <= 8 for n in res.meta["n_cbv_selected"].values())
    for q in (1, 2):
        assert np.median(res.flux[res.quarter == q]) == pytest.approx(1.0, abs=1e-4)
    assert d.label == "arc2(n_max=8)"


def test_arc2_registry_and_modes():
    d = sx.make_detrender({"name": "arc2", "n_max": 6})
    assert isinstance(d, sx.ARC2) and d.n_max == 6 and not d.linear
    with pytest.raises(ValueError):
        sx.ARC2(backend="matlab")
    obs, loader, _ = _obs_with_cbv_trend()
    with pytest.raises(ValueError):
        sx.ARC2(loader=loader)(obs, mode="marginalize")


def test_oxksc_missing_gives_helpful_error(monkeypatch, tmp_path):
    monkeypatch.delenv(_oxksc.ENV_VAR, raising=False)
    for name in [m for m in sys.modules if m == "oxksc" or m.startswith("oxksc.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "OxKeplerSC" not in p])
    if _oxksc.available():
        pytest.skip("an installed oxksc package is importable")
    with pytest.raises(ImportError, match="OXKEPLERSC_PATH"):
        _oxksc.load_oxksc_cbv()
    with pytest.raises(ImportError, match="does not look like"):
        _oxksc.load_oxksc_cbv(tmp_path)


def test_scipy_aliases_are_temporary():
    import scipy
    had = hasattr(scipy, "log")
    with _oxksc.scipy_aliases():
        assert scipy.log is np.log or had
    assert hasattr(scipy, "log") == had


# ---------------------------------------------------------------- cross-checks against OxKeplerSC

@needs_oxksc
def test_native_matches_oxksc_and_2017_reference():
    from astropy.io import fits

    root = os.environ.get(_oxksc.ENV_VAR)
    if not root or not os.path.isdir(os.path.join(root, "examples")):
        pytest.skip("OxKeplerSC example data not found")
    ex = os.path.join(root, "examples")
    inp = fits.open(os.path.join(ex, "kplr009579641-2010174085026_llc.fits"))
    ref = fits.open(os.path.join(ex, "kplr009579641-2010174085026_llc_cbvshrink.fits"))
    cbvf = fits.open(os.path.join(ex, "kplr2010174085026-q05-d25_lcbv.fits"))
    mod, out = inp[0].header["MODULE"], inp[0].header["OUTPUT"]
    cbv = np.column_stack([cbvf[f"MODOUT_{mod}_{out}"].data[f"VECTOR_{j}"].astype(float)
                           for j in range(1, 17)])
    sap = inp[1].data["SAP_FLUX"].astype(float)
    ok = np.isfinite(sap)

    fit = select_n_cbv(sap, cbv, 8)
    ox = _oxksc.load_oxksc_cbv()
    with _oxksc.scipy_aliases():
        (nb, _, _), (multi, wmulti) = ox.sel_nb(sap, cbv, nBmax=8)
    assert fit.n_cbv == nb
    for k in range(8):
        np.testing.assert_allclose(fit.weights_all[k, :k + 1], wmulti[k, :k + 1], rtol=1e-9)
    np.testing.assert_allclose(fit.corrected_all[:, ok], multi[:, ok], rtol=1e-12)
    # The repo's stored output (float32) from 2017.
    for k in range(8):
        r = ref[1].data[f"CBV{k + 1}_FLUX"].astype(float)
        np.testing.assert_allclose(fit.corrected_all[k, ok], r[ok], rtol=1e-6)
        wref = [float(ref[1].header[f"CBVW_{k + 1}{i}"]) for i in range(k + 1)]
        np.testing.assert_allclose(fit.weights_all[k, :k + 1], wref, rtol=1e-9)


@needs_oxksc
def test_detrender_backends_agree():
    obs, loader, _ = _obs_with_cbv_trend(seed=5)
    a = sx.ARC2(loader=loader)(obs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sx.ARC2(loader=loader, backend="oxksc")(obs)
    assert a.meta["n_cbv_selected"] == b.meta["n_cbv_selected"]
    np.testing.assert_allclose(a.flux, b.flux, rtol=1e-12)
