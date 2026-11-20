"""ARC2 CBV correction (Aigrain et al. 2017), reimplemented in plain numpy.

ARC2 ("Astrophysically Robust Correction 2"; Aigrain, Parviainen, Roberts,
Reece & Evans 2017, MNRAS 471, 759) fits the Kepler CBVs to each light curve
with variational-Bayes linear regression under automatic-relevance-
determination (ARD) "shrinkage" priors (Roberts et al. 2013, MNRAS 435,
3639; the update equations are those of Drugowitsch 2013, "Variational
Bayesian inference for linear and logistic regression"). It fits the first
1, 2, ..., ``n_max`` CBVs and keeps the smallest number that brings the
light curve's 5-95 percentile range down to the typical level, then backs
off while the correction adds more than 10% point-to-point noise.

This is an independent MIT-licensed implementation. The original GPL-3 code
(OxKeplerSC, https://github.com/OxES/OxKeplerSC) can be used instead through
``ARC2(backend="oxksc")``; see :mod:`systematix._oxksc`. The tests check
that both give the same weights and corrected flux.

Only the CBV correction is reproduced here, not ARC2's jump correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

__all__ = ["vb_linear_ard", "fit_cbv_weights", "medransig", "select_n_cbv", "ARC2Fit"]


def vb_linear_ard(X, y, a0=1e-2, b0=1e-4, c0=1e-2, d0=1e-4, max_iter=500, tol=1e-5) -> dict:
    """Variational-Bayes linear regression ``y ~ X w`` with an ARD prior per weight.

    Model: ``y | w, tau ~ N(X w, 1/tau)``, ``w_i | tau, alpha_i ~ N(0, 1/(tau alpha_i))``,
    ``tau ~ Gamma(a0, b0)``, ``alpha_i ~ Gamma(c0, d0)``. Iterates the
    mean-field updates until the variational bound changes by less than
    ``tol`` (relative).

    Returns ``{"w", "V", "E_alpha", "E_tau", "bound", "n_iter"}`` where ``V``
    is the weight covariance divided by the noise variance.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N, D = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    an = a0 + N / 2.0
    cn = c0 + 0.5
    E_alpha = np.full(D, c0 / d0)
    L_last = -np.inf
    for it in range(max_iter):
        invV = XtX + np.diag(E_alpha)
        cho = np.linalg.cholesky(invV)
        V = np.linalg.inv(invV)
        logdetV = -2.0 * np.sum(np.log(np.diag(cho)))
        w = V @ Xty
        resid = X @ w - y
        sse = resid @ resid
        bn = b0 + 0.5 * (sse + np.sum(w * w * E_alpha))
        E_tau = an / bn
        dn = d0 + 0.5 * (E_tau * w * w + np.diag(V))
        E_alpha = cn / dn
        L = (-0.5 * (E_tau * sse + np.sum(X * (X @ V))) + 0.5 * logdetV - b0 * E_tau
             + gammaln(an) - an * np.log(bn) + an + D * gammaln(cn) - cn * np.sum(np.log(dn)))
        if L < L_last:
            raise RuntimeError(f"variational bound decreased ({L_last:.6g} -> {L:.6g})")
        if abs(L_last - L) < abs(tol * L):
            break
        L_last = L
    L += (-0.5 * (N * np.log(2 * np.pi) - D) - gammaln(a0) + a0 * np.log(b0)
          + D * (-gammaln(c0) + c0 * np.log(d0)))
    return {"w": w, "V": V, "E_alpha": E_alpha, "E_tau": E_tau, "bound": L, "n_iter": it + 1}


def fit_cbv_weights(flux, basis) -> np.ndarray:
    """ARD weights of ``basis`` columns (nobs x nb) for one light curve, in flux units.

    The basis is divided by its overall standard deviation and the flux is
    standardized before the fit; a constant column absorbs the mean.
    """
    flux = np.asarray(flux, dtype=float)
    B = np.asarray(basis, dtype=float)
    Bs = B.std()
    X = np.column_stack([B / Bs, np.ones(B.shape[0])])
    Fm, Fs = flux.mean(), flux.std()
    fit = vb_linear_ard(X, (flux - Fm) / Fs)
    return fit["w"][:-1] * Fs / Bs


def medransig(flux) -> tuple[float, float, float]:
    """Median, 5-95 percentile range (ppm) and point-to-point scatter (ppm) of a light curve."""
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(flux)
    med = np.median(flux[ok])
    norm = flux / med
    srt = np.sort(norm)                       # NaNs sort to the end
    n = int(ok.sum())
    ran = srt[int(n * 0.95)] - srt[int(n * 0.05)]
    d = np.diff(norm)
    sig = 1.48 * np.median(np.abs(d[np.isfinite(d)]))
    return float(med), float(ran * 1e6), float(sig * 1e6)


@dataclass
class ARC2Fit:
    n_cbv: int                  # selected number of CBVs
    corrected: np.ndarray       # corrected flux with n_cbv CBVs (median restored)
    weights: np.ndarray         # (n_cbv,)
    corrected_all: np.ndarray   # (n_max, nobs): corrected flux for 1..n_max CBVs
    weights_all: np.ndarray     # (n_max, n_max): row k holds the k+1 weights
    ranges: np.ndarray          # 5-95% range after each correction, ppm
    sigmas: np.ndarray          # point-to-point scatter after each correction, ppm


def select_n_cbv(flux, cbv, n_max: int = 8, use=None) -> ARC2Fit:
    """Fit 1..``n_max`` CBVs and pick the number ARC2's rule selects.

    ``cbv`` is (nobs x >= n_max), columns in CBV order. ``use`` masks the
    cadences used in the fit (NaNs are always excluded); the correction is
    evaluated on all cadences.
    """
    flux = np.asarray(flux, dtype=float)
    cbv = np.asarray(cbv, dtype=float)[:, :n_max]
    n_max = cbv.shape[1]
    ok = np.isfinite(flux)
    if use is not None:
        ok &= np.asarray(use, dtype=bool)
    med_raw, _, sig_raw = medransig(flux[ok])

    corrected_all = np.zeros((n_max, flux.size))
    weights_all = np.zeros((n_max, n_max))
    ranges = np.zeros(n_max)
    sigmas = np.zeros(n_max)
    for k in range(n_max):
        w = fit_cbv_weights(flux[ok], cbv[ok, :k + 1])
        weights_all[k, :k + 1] = w
        c = flux - cbv[:, :k + 1] @ w
        med, ranges[k], sigmas[k] = medransig(c[ok])
        corrected_all[k] = c - med + med_raw

    # Smallest number of CBVs whose range is no longer an outlier above the rest...
    med_ran = np.median(ranges)
    sig_ran = 1.48 * np.median(np.abs(ranges - med_ran))
    j = int(np.flatnonzero(ranges < med_ran + 3 * sig_ran)[0])
    # ...backing off while the correction inflates point-to-point noise by > 10%.
    while sigmas[j] > 1.1 * sig_raw and j > 0:
        j -= 1
    return ARC2Fit(j + 1, corrected_all[j].copy(), weights_all[j, :j + 1].copy(),
                   corrected_all, weights_all, ranges, sigmas)
