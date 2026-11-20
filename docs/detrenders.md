# Detrenders

All detrenders share one interface: call one on an
{class}`~systematix.observation.Observation` and get a
{class}`~systematix.detrend.DetrendResult` back.

```python
res = sx.CBV(n_cbv=8)(obs)                         # fit and subtract a trend
res = sx.Poly(order=2)(obs, mode="marginalize")    # or return a nuisance model

res.time, res.flux, res.flux_err    # relative flux, ~1
res.trend                           # the subtracted trend (subtract mode)
res.full()                          # flux on the input cadences, NaN where dropped
```

## How detrending works

Every detrender first divides each quarter by its median, giving the relative flux

$$
r(t) = \frac{F(t)}{\widetilde{F}_q} - 1 .
$$

Then it does one of two things.

`mode="subtract"`
: Estimate a trend $T(t)$ and return $1 + r(t) - T(t)$, optionally sigma clipped.
  Every detrender supports this mode.

`mode="marginalize"`
: Linear detrenders only. Return $1 + r(t)$ untouched, plus a design matrix $M$
  and a Gaussian prior on its weights (`prior_mu`, `prior_sigma`). A likelihood can
  then integrate the weights out instead of fixing them. `prior_sigma = inf` means a
  flat prior. Per-quarter offset columns are always included, so the median
  normalization does not matter in this mode.

Linear detrenders fit their weights by generalized least squares with the Gaussian
prior, which is ridge regression when `prior_sigma` is finite. Trends are subtracted
in relative flux, not divided out; for percent-level signals the difference is
second order.

## The detrenders

| Detrender | What it removes | Modes |
|---|---|---|
| {class}`QuarterMedian() <systematix.detrend.QuarterMedian>` | nothing beyond each quarter's median (what lightkurve's `stitch()` does) | subtract |
| {class}`RollingMedian(window=10) <systematix.detrend.RollingMedian>` | a running median over `window` days, within each quarter | subtract |
| {class}`LkFlatten(window=2, polyorder=2) <systematix.detrend.LkFlatten>` | lightkurve's iterative Savitzky–Golay `flatten`, window in days | subtract |
| {class}`Poly(order=2) <systematix.detrend.Poly>` | a polynomial per quarter (`order=0`: free offsets only) | subtract, marginalize |
| {class}`Ramps(taus=(0.5, 2), order=1) <systematix.detrend.Ramps>` | exponential thermal-recovery ramps after each gap, plus a per-quarter polynomial | subtract, marginalize |
| {class}`CBV(n_cbv=8, prior_sigma=None) <systematix.detrend.CBV>` | Kepler DR25 CBVs per quarter, Gaussian prior on the weights | subtract, marginalize |
| {class}`ARC2(n_max=8) <systematix.detrend.ARC2>` | ARC2's CBV correction: shrinkage-prior fit, number of CBVs chosen per quarter | subtract |

Every detrender also takes `clip` and `clip_method` (see {ref}`clipping`).

### Ramps

After each data gap, Kepler's focal plane warms or cools and the flux recovers
along an exponential. {class}`~systematix.detrend.Ramps` adds one column
$\exp(-(t - t_0)/\tau)$ per segment start $t_0$ and timescale $\tau$, running until
the next segment starts. Segments start at each quarter's first cadence and after
every gap longer than `min_gap` days (default 0.5). `prior_sigma` sets a Gaussian
prior on the ramp amplitudes in relative flux (default flat).

### CBV

{class}`~systematix.detrend.CBV` fits the first `n_cbv` DR25 cotrending basis
vectors of the star's CCD output, separately in each quarter (a block-diagonal
design matrix). It needs `MODULE` and `OUTPUT` in `quarter_meta`, which
{meth}`Observation.from_systematics <systematix.observation.Observation.from_systematics>`
provides. The CBV files are downloaded from MAST once (~25 MB per quarter) and
cached in `$SYSTEMATIX_CACHE/cbv`. Rows are matched on `cadenceno`; cadences missing
from the CBV table, or flagged as gaps there, get zero rows.

`prior_sigma` is the Gaussian prior width on each CBV weight, in relative flux per
unit-norm vector. With the default `None` the fit is plain least squares, which
overfits. The CBVs are zero-mean and roughly unit-norm over a quarter, so a trend of
rms $s$ over $N$ cadences needs weights of about $s\sqrt{N}$.
`examples/transfer_sweep.py` sets the width from flat-prior fits to the donor's own
SAP flux.

(arc2)=
### ARC2

{class}`~systematix.detrend.ARC2` is an MIT reimplementation of the CBV correction
in ARC2 (Aigrain et al. 2017, MNRAS 471, 759;
[arXiv:1706.03064](https://arxiv.org/abs/1706.03064)). For each quarter it fits the
first 1, 2, ..., `n_max` CBVs by variational-Bayes linear regression with an
automatic-relevance-determination ("shrinkage") prior on each weight. It keeps the
smallest number that brings the light curve's 5–95 % range down to the typical
level, then backs off while the correction adds more than 10 % point-to-point
noise.

```python
res = sx.ARC2(n_max=8)(obs)
res.meta["n_cbv_selected"]      # {quarter: number of CBVs kept}
```

ARC2 is nonlinear, so it supports subtract mode only. ARC2's jump correction is not
included.

`backend="oxksc"` calls the original GPL-3 code,
[OxKeplerSC](https://github.com/OxES/OxKeplerSC), instead (see {doc}`install`).
systematix looks for an importable `oxksc` package first, then a clone at
`$OXKEPLERSC_PATH` (or the `oxksc_path` argument). OxKeplerSC calls `scipy.log` and
`scipy.multiply`, which current scipy removed; systematix adds them back for the
duration of each call only. The test suite checks that both backends give the same
weights to 1e-9 relative or better.

(clipping)=
## Sigma clipping

Any detrender takes `clip=k` to drop outliers after subtraction:

- `clip_method="mad"` (default): iterative, up to three passes, keeping points
  within $k$ times $1.4826\,\mathrm{MAD}$ of the median.
- `clip_method="std"`: one pass of mean ± $k$ standard deviations, as in spotgp's
  `scripts/injection_recovery.py`.

`res.mask` marks the input cadences that were kept.

## Building detrenders from configuration

{func}`~systematix.detrend.make_detrender` builds a detrender from a name or a
dictionary, so a run can be described in a config file:

```python
sx.make_detrender("quarter_median")
sx.make_detrender({"name": "cbv", "n_cbv": 8, "prior_sigma": 0.05, "clip": 5})
```

The names are the keys of `sx.DETRENDERS`: `quarter_median`, `rolling_median`,
`lk_flatten`, `poly`, `ramps`, `cbv` and `arc2`. A `mode` key in the dictionary is
ignored; pass the mode when you call the detrender.

Each detrender has a `label`, such as `cbv(n_cbv=8, prior_sigma=0.05, order=0)`,
that {func}`~systematix.transfer.transfer_sweep` uses to name its rows.

## Using the result

{class}`~systematix.detrend.DetrendResult` holds `time`, `flux`, `flux_err`,
`quarter`, `mask` and `meta`, plus `trend` (subtract mode) or `design`,
`prior_mu` and `prior_sigma` (marginalize mode).

`res.full(values=None)`
: Put `flux` (or any array of kept-cadence values) back on the input cadences,
  with NaN where cadences were dropped.

`res.binned(dt)`
: Inverse-variance average onto bins of `dt` days within each quarter. Design-matrix
  rows are averaged with the same weights, so a linear nuisance model stays exact
  after binning.

`res.gp_kwargs()`
: For marginalize mode: `{"offsets": design, "offset_prior": (prior_mu, prior_sigma)}`,
  the keyword arguments spotgp's `GPSolver` takes to integrate the nuisance columns
  out of its likelihood.

```python
res = sx.Ramps(taus=(0.5, 2.0), order=1)(obs, mode="marginalize").binned(0.25)
solver = GPSolver(data, model, **res.gp_kwargs())    # spotgp
```

## Writing your own

Subclass {class}`~systematix.detrend.Detrender` and implement `_trend`, or subclass
{class}`~systematix.detrend.LinearDetrender` and implement `_design`:

```python
import numpy as np
from systematix.detrend import LinearDetrender, segment_poly_design, DETRENDERS

class Sinusoids(LinearDetrender):
    """Per-quarter offsets plus fixed-period sinusoids (e.g. a known reaction-wheel period)."""

    name = "sinusoids"

    def __init__(self, periods=(3.2,), clip=None, clip_method="mad"):
        super().__init__(clip, clip_method)
        self.periods = tuple(periods)

    def params(self):
        return {"periods": self.periods, **super().params()}

    def _design(self, obs):
        base = segment_poly_design(obs.time, obs.quarter, order=0)
        cols = [f(2 * np.pi * obs.time / P) for P in self.periods for f in (np.sin, np.cos)]
        M = np.column_stack([base, *cols])
        return M, np.zeros(M.shape[1]), np.full(M.shape[1], np.inf)   # flat priors

DETRENDERS["sinusoids"] = Sinusoids    # optional: make it available to make_detrender
```

`_trend(obs, rel, err, good)` receives the relative flux and its error for all
cadences, plus a mask of good ones, and returns the trend in relative flux.
`_design(obs)` returns `(M, prior_mu, prior_sigma)` over all cadences; columns that
are zero on every kept cadence are dropped automatically.
