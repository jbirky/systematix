"""Mode (c): a marginalize-mode DetrendResult handed to spotgp's GPSolver.

Skipped when spotgp is not importable.
"""

import warnings

import numpy as np
import pytest

import systematix as sx

spotgp = pytest.importorskip("spotgp")
jnp = pytest.importorskip("jax.numpy")

HPARAM = dict(peq=5.0, kappa=0.1, inc=np.pi / 3, lspot=4.0, tau_spot=2.0, sigma_k=5e-3)


def smooth_basis(n, k, seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, k)).cumsum(axis=0)
    V -= V.mean(axis=0)
    return V / np.linalg.norm(V, axis=0)


@pytest.fixture
def setup():
    tmpl = sx.synthetic_cadence(n_quarters=2, quarter_days=20.0, dt=0.25, gap_every=10.0,
                                gap_days=1.0, sigma=1e-4, seed=0)
    tmpl = tmpl.replace(quarter_meta={q: {"MODULE": 2, "OUTPUT": 1} for q in tmpl.quarters})
    full = np.arange(tmpl.cadenceno.min(), tmpl.cadenceno.max() + 1)
    vecs = {q: smooth_basis(full.size, 16, 10 + q) for q in tmpl.quarters}

    def loader(q, module, output):
        return {"cadenceno": full, "vectors": vecs[q], "gap": np.zeros(full.size, bool)}

    det = sx.CBV(n_cbv=3, prior_sigma=0.05, loader=loader)
    V = det._cbv_matrix(tmpl, 3)
    w_true = np.array([0.03, -0.02, 0.01, -0.025, 0.015, 0.0])
    rng = np.random.default_rng(1)
    star = 5e-3 * np.sin(2 * np.pi * tmpl.time / 5.0)
    flux = 1e4 * np.where(tmpl.quarter == tmpl.quarters[0], 1.0, 1.3) * (1 + star + V @ w_true
                                                                       + rng.normal(0, 1e-4, len(tmpl)))
    obs = tmpl.replace(flux=flux, flux_err=1e4 * 1e-4 * np.where(tmpl.quarter == tmpl.quarters[0], 1.0, 1.3))
    return obs, det, w_true


def make_gp(res, **extra):
    data = spotgp.TimeSeriesData(res.time, res.flux, res.flux_err)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spotgp.GPSolver(data, model_or_hparam=dict(HPARAM), matrix_solver="cholesky_banded",
                               **extra)


def test_gp_kwargs_requires_marginalize(setup):
    obs, det, _ = setup
    with pytest.raises(ValueError):
        det(obs).gp_kwargs()
    kw = det(obs, mode="marginalize").gp_kwargs()
    assert set(kw) == {"offsets", "offset_prior"}
    mu, sigma = kw["offset_prior"]
    assert np.sum(np.isinf(sigma)) == 2 and np.all(sigma[np.isfinite(sigma)] == 0.05)


def test_cbv_weights_marginalized_in_gp(setup):
    obs, det, w_true = setup
    res = det(obs, mode="marginalize")
    gp = make_gp(res, **res.gp_kwargs())
    ll = float(gp.log_likelihood_fn(jnp.asarray(gp.theta0)))
    assert np.isfinite(ll)
    a_hat, a_cov = gp.offset_estimates(return_cov=True)
    # Columns: 2 flat quarter offsets, then 3 CBVs per quarter (relative-flux weights).
    cbv_hat = a_hat[2:]
    cbv_err = np.sqrt(np.diag(a_cov))[2:]
    pulls = (cbv_hat - w_true) / cbv_err
    assert np.all(np.abs(pulls) < 5), pulls


def test_marginal_beats_ignoring_the_systematics(setup):
    """The data favour modelling the injected CBV trend over ignoring it."""
    obs, det, _ = setup
    res = det(obs, mode="marginalize")
    with_cbv = make_gp(res, **res.gp_kwargs())
    offsets_only = make_gp(res, offsets=res.design[:, :2])
    ll_cbv = float(with_cbv.log_likelihood_fn(jnp.asarray(with_cbv.theta0)))
    ll_off = float(offsets_only.log_likelihood_fn(jnp.asarray(offsets_only.theta0)))
    assert ll_cbv > ll_off + 10
