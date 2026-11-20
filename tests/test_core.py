import warnings

import numpy as np
import pytest
from astropy.time import Time
from astropy.utils.masked import Masked
import astropy.units as u

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightkurve as lk

import systematix as sx
from systematix import core
from systematix.cli import main, to_table


# ---------------------------------------------------------------- helpers

def pdc_forward(f_cot, crowdsap, flfrcsap):
    """PDC's crowding + flux-fraction step (Kinemuchi et al. 2012)."""
    return (f_cot - (1 - crowdsap) * np.nanmedian(f_cot)) / flfrcsap


def fake_quarter(quarter, t0, n=500, crowdsap=0.9, flfrcsap=0.85, seed=0, nan_idx=()):
    """A fake Kepler file with known star signal and additive systematics."""
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n) * 0.0204
    level = 1e5 * (1 + 0.1 * quarter)
    star = level * (1 + 0.01 * np.sin(2 * np.pi * t / 7.3)) + rng.normal(0, 30, n)
    sys = 400 * np.exp(-(t - t0) / 3) + 50 * (t - t0) / t[-1]  # thermal ramp + drift
    sap = star + sys
    pdc = pdc_forward(star, crowdsap, flfrcsap)
    sap[list(nan_idx)] = np.nan
    pdc[list(nan_idx)] = np.nan

    lc = lk.LightCurve(
        time=Time(t, format="bkjd", scale="tdb"),
        flux=pdc * u.electron / u.s,
        flux_err=np.full(n, 30.0) * u.electron / u.s,
    )
    lc["sap_flux"] = sap * u.electron / u.s
    lc["sap_flux_err"] = np.full(n, 30.0) * u.electron / u.s
    lc["pdcsap_flux"] = pdc * u.electron / u.s
    lc["quality"] = np.zeros(n, dtype=np.int32)
    lc["cadenceno"] = np.arange(n) + 1000 * quarter
    lc.meta.update(KEPLERID=123, OBJECT="KIC 123", LABEL="KIC 123", QUARTER=quarter,
                   CROWDSAP=crowdsap, FLFRCSAP=flfrcsap)
    return lc, sys, star


# ---------------------------------------------------------------- parse_kic

@pytest.mark.parametrize("arg", [11904151, "11904151", "KIC 11904151", "kplr011904151", np.int64(11904151)])
def test_parse_kic(arg):
    assert sx.parse_kic(arg) == 11904151


@pytest.mark.parametrize("arg", ["KIC", "TIC 1 2", ""])
def test_parse_kic_bad(arg):
    with pytest.raises(ValueError):
        sx.parse_kic(arg)


# ---------------------------------------------------------------- compute_systematics

def test_recovers_injected_systematics_exactly():
    lc, sys_true, _ = fake_quarter(3, 200.0)
    sys, pdc_rescaled = sx.compute_systematics(
        lc["sap_flux"], lc["pdcsap_flux"], crowdsap=0.9, flfrcsap=0.85
    )
    np.testing.assert_allclose(sys, sys_true, atol=1e-6)
    np.testing.assert_allclose(pdc_rescaled, lc["sap_flux"].value - sys_true, rtol=1e-12)


def test_literal_difference_leaks_star_signal():
    """Without undoing crowding, SAP-PDCSAP has an offset and leaks (1-1/f) of the star."""
    lc, sys_true, star = fake_quarter(3, 200.0)
    sys, _ = sx.compute_systematics(lc["sap_flux"], lc["pdcsap_flux"], 0.9, 0.85, undo_crowding=False)
    np.testing.assert_allclose(sys, lc["sap_flux"].value - lc["pdcsap_flux"].value)
    resid = sys - sys_true
    # The residual is a scaled copy of the star plus a constant.
    slope = np.polyfit(star, resid, 1)[0]
    assert slope == pytest.approx(1 - 1 / 0.85, rel=1e-6)


def test_uncrowded_target_literal_equals_corrected():
    lc, _, _ = fake_quarter(1, 100.0, crowdsap=1.0, flfrcsap=1.0)
    a, _ = sx.compute_systematics(lc["sap_flux"], lc["pdcsap_flux"], 1.0, 1.0)
    b, _ = sx.compute_systematics(lc["sap_flux"], lc["pdcsap_flux"], 1.0, 1.0, undo_crowding=False)
    np.testing.assert_allclose(a, b)


def test_compute_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sx.compute_systematics(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        sx.compute_systematics(np.ones(3), np.ones(3), crowdsap=0.0)
    with pytest.raises(ValueError):
        sx.compute_systematics(np.ones(3), np.ones(3), flfrcsap=1.2)


# ---------------------------------------------------------------- systematics_from_lightcurve

def test_from_lightcurve_columns_and_units():
    lc, sys_true, _ = fake_quarter(5, 300.0)
    out = sx.systematics_from_lightcurve(lc)
    assert out.flux.unit == u.electron / u.s
    for col in ("sap_flux", "sap_flux_err", "pdcsap_flux", "pdcsap_flux_rescaled", "quality", "cadenceno", "quarter"):
        assert col in out.colnames
    assert np.all(out["quarter"] == 5)
    assert np.all(np.isnan(out.flux_err.value))
    np.testing.assert_allclose(out.flux.value, sys_true, atol=1e-6)


def test_from_lightcurve_normalize():
    lc, sys_true, _ = fake_quarter(5, 300.0)
    out = sx.systematics_from_lightcurve(lc, normalize=True)
    assert out.flux.unit == u.dimensionless_unscaled
    np.testing.assert_allclose(out.flux.value, sys_true / np.median(lc["sap_flux"].value), atol=1e-10)
    # raw columns stay in e-/s
    assert out["sap_flux"].unit == u.electron / u.s


def test_nan_cadences_dropped_or_kept():
    nan_idx = (0, 10, 11, 499)
    lc, sys_true, _ = fake_quarter(2, 150.0, nan_idx=nan_idx)
    dropped = sx.systematics_from_lightcurve(lc)
    kept = sx.systematics_from_lightcurve(lc, dropna=False)
    assert len(dropped) == 500 - len(nan_idx)
    assert len(kept) == 500
    assert np.isnan(kept.flux.value[list(nan_idx)]).all()
    good = np.setdiff1d(np.arange(500), nan_idx)
    np.testing.assert_allclose(dropped.flux.value, sys_true[good], atol=1e-6)


def test_masked_columns_are_handled():
    """lightkurve returns masked quantities when the FITS column has NaNs."""
    lc, sys_true, _ = fake_quarter(4, 250.0)
    mask = np.zeros(len(lc), bool)
    mask[[3, 7]] = True
    # Underlying values stay finite, so only the mask marks them bad.
    lc["pdcsap_flux"] = Masked(lc["pdcsap_flux"].copy(), mask=mask)
    out = sx.systematics_from_lightcurve(lc)
    assert len(out) == len(lc) - 2
    assert np.isfinite(out.flux.value).all()
    # Dropping cadences shifts median(PDCSAP) slightly -> constant offset only.
    resid = np.asarray(out.flux.value) - np.delete(sys_true, [3, 7])
    assert np.std(resid) < 1e-6 and abs(np.mean(resid)) < 1.0

    # plain numpy masked arrays too
    sys, _ = sx.compute_systematics(
        np.ma.MaskedArray(lc["sap_flux"].value, mask=mask), lc["pdcsap_flux"].unmasked, 0.9, 0.85
    )
    assert np.isnan(sys[[3, 7]]).all() and np.isfinite(np.delete(sys, [3, 7])).all()


# ---------------------------------------------------------------- get_systematics (offline)

@pytest.fixture
def fake_download(monkeypatch):
    params = {1: (0.95, 0.9), 2: (0.80, 0.7), 3: (1.0, 0.99)}
    truths = {}
    calls = {}

    def _download(kic, quarters=None, cadence="long", quality_bitmask="default", download_dir=None):
        calls.update(kic=kic, quarters=quarters, cadence=cadence)
        qs = [quarters] if isinstance(quarters, int) else (quarters or sorted(params))
        lcs = []
        for i, q in enumerate(qs):
            lc, sys, _ = fake_quarter(q, 100.0 * q, crowdsap=params[q][0], flfrcsap=params[q][1], seed=q)
            truths[q] = sys
            lcs.append(lc)
        return lk.LightCurveCollection(lcs[::-1])  # out of order on purpose

    monkeypatch.setattr(core, "download_lightcurves", _download)
    return truths, calls, params


def test_get_systematics_stitches_quarters(fake_download):
    truths, calls, params = fake_download
    out = sx.get_systematics("KIC 123")
    assert calls["kic"] == 123
    assert len(out) == 1500
    assert np.all(np.diff(out.time.value) > 0)
    assert out.meta["QUARTERS"] == [1, 2, 3]
    assert out.meta["CROWDSAP"] == {q: params[q][0] for q in params}
    for q, sys in truths.items():
        np.testing.assert_allclose(out.flux.value[out["quarter"] == q], sys, atol=1e-6)


def test_get_systematics_normalized_per_quarter(fake_download):
    truths, _, _ = fake_download
    out = sx.get_systematics(123, quarters=[2, 3], normalize=True)
    assert out.meta["QUARTERS"] == [2, 3]
    for q in (2, 3):
        sel = out["quarter"] == q
        expect = truths[q] / np.median(out["sap_flux"].value[sel])
        np.testing.assert_allclose(out.flux.value[sel], expect, atol=1e-10)


def test_get_systematics_single_quarter(fake_download):
    truths, _, params = fake_download
    out = sx.get_systematics(123, quarters=2)
    assert out.meta["QUARTERS"] == [2]
    assert out.meta["CROWDSAP"] == {2: params[2][0]}
    np.testing.assert_allclose(out.flux.value, truths[2], atol=1e-6)


def test_cli_writes_table(fake_download, tmp_path, capsys):
    path = tmp_path / "sys.ecsv"
    png = tmp_path / "sys.png"
    assert main(["123", "-q", "1", "2", "-o", str(path), "--plot", str(png)]) == 0
    from astropy.table import Table

    t = Table.read(path)
    assert len(t) == 1000
    assert {"time", "systematics", "sap_flux", "pdcsap_flux", "quarter"} <= set(t.colnames)
    assert t.meta["KEPLERID"] == 123
    assert png.exists() and png.stat().st_size > 0


def test_download_rejects_bad_cadence():
    with pytest.raises(ValueError):
        core.download_lightcurves(123, cadence="medium")


# ---------------------------------------------------------------- live MAST

@pytest.mark.remote
def test_kepler10_live(tmp_path):
    out = sx.get_systematics(11904151, quarters=[2, 3], download_dir=str(tmp_path))
    assert set(np.unique(out["quarter"])) == {2, 3}
    assert out.meta["KEPLERID"] == 11904151
    np.testing.assert_allclose(
        out.flux.value, out["sap_flux"].value - out["pdcsap_flux_rescaled"].value
    )
    # Undoing crowding removes the large constant offset of the literal difference.
    literal = sx.get_systematics(11904151, quarters=[3], undo_crowding=False, download_dir=str(tmp_path))
    corrected = out[out["quarter"] == 3]
    assert abs(np.median(corrected.flux.value)) < 0.1 * abs(np.median(literal.flux.value))
