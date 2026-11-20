# Generated systematics (S3)

Sources S1 and S2 reuse one real star's systematics. Source S3,
{class}`~systematix.generated.Generated`, draws *new* systematics for any star's
cadence, from the systematics of many quiet donor stars on the same CCD output. It
is built in three stages:

1. **Decompose** each donor's SAP flux into a linear systematics model
   ({mod}`systematix.decompose`), splitting quarters at pointing steps
   ({mod}`systematix.steps`), and measure the residual noise spectrum
   ({mod}`systematix.rednoise`).
2. **Collect** the donors' weights for each (quarter, CCD channel) into a
   {class}`~systematix.generated.WeightModel`, saved as `<dir>/Qqq/chCC.h5`.
3. **Draw** weights from that model and evaluate the same design matrix on a
   template star's cadence ({class}`~systematix.generated.Generated`).

Stages 1 and 2 run once over the donor library. Stage 3 runs per injection.

```python
import systematix as sx

lc = sx.get_systematics(kic, quarters=[2, 3, 4, 5])
template = sx.Observation.from_systematics(lc)       # cadence, errors, header values

gen = sx.Generated(template, "/path/to/models")      # reads Qqq/chCC.h5 for each quarter
sys = gen.realize(rng=1)
obs = sx.inject(sys, f_spot, rng=2)
```

```{important}
The weight models are not shipped with the package. They are built from the donor
library by the Phase 4 production scripts in `validation/phase4_production/`
(`decompose.py`, then `weights.py`).
```

## The systematics model

{func}`~systematix.decompose.design_blocks` builds a design matrix $M$ with four
blocks, all per quarter:

| Block | Columns | Defaults |
|---|---|---|
| `offsets` | one indicator per segment; quarters are split into segments at pointing steps | |
| `polynomial` | centred, scaled powers of time per segment | linear and quadratic (`POLY_ORDER = 2`); none for segments shorter than 48 cadences (1 day) |
| `ramps` | $\exp(-(t - t_0)/\tau)$ after each quarter start and each gap longer than `MIN_GAP` | $\tau$ = 0.5 and 2 d (`TAUS`); `MIN_GAP` = 0.5 d |
| `cbv` | the first `N_CBV` DR25 CBVs of the star's output | `N_CBV = 8` |

The relative systematics are $s(t) = M w$, and a realization's flux scale is
$L_q\,(1 + s(t))$, normalized so that $s$ has median 0 in each quarter.

### Pointing steps

Kepler's pointing occasionally jumped, which changes the flux in an aperture
abruptly. {func}`~systematix.steps.pointing_steps` finds these jumps in one star's
`POS_CORR1`/`POS_CORR2` columns: a jump of more than 0.02 px between adjacent good
cadences, or across a gap, comparing the median positions over 0.5 d on either side.
Jumps in the first day of a quarter are ignored and jumps within half a day merge.
{func}`~systematix.steps.consensus_steps` keeps the step times found in at least
half the stars on an output, so every star on that output gets the same segments.
{func}`~systematix.steps.segment_labels` then labels each cadence
`100 * quarter + k`.

### Fitting a donor

{func}`~systematix.decompose.fit_flat` fits all blocks together with flat priors, by
generalized least squares, adding a jitter term so the reduced χ² is 1.
{func}`~systematix.decompose.keyed_weights` then re-expresses one quarter's weights
on keys that mean the same thing for every star on an output:

| Key | Meaning |
|---|---|
| `("offset", k)` | segment *k*'s offset relative to the quarter's first segment |
| `("poly", k, p)` | power *p* of the polynomial in segment *k* |
| `("ramp", t0, tau)` | a ramp, with $t_0$ snapped to the output's canonical gap-end times ({func}`~systematix.decompose.canonical_t0`) |
| `("cbv", j)` | CBV *j* |

Keys are stored as strings such as `"ramp:200.3:0.5"`
({func}`~systematix.decompose.key_str`, {func}`~systematix.decompose.parse_key`).

### Residual noise

The fit leaves each donor a residual in relative flux. Its periodogram is flat at
high frequency, where only white noise remains, and rises at low frequency from
granulation, low-level variability and systematics the model does not capture.

The fit itself removes most of the lowest-frequency power, so the residual
underestimates the noise that was there before the fit. Noise drawn with the
residual spectrum and then fit again would lose that power twice.
{func}`~systematix.rednoise.prefit_red` corrects for this. For each output and
quarter it measures the fit's power transfer $T(f)$ on real donor designs, estimates
each donor's pre-fit excess over white noise, and fits it as an amplitude times a
shared shape. Because stronger red noise is also steeper, donors are split into
groups of similar amplitude, each with its own shape and transfer.

{func}`~systematix.rednoise.red_noise` draws a stationary Gaussian series with a
given binned spectrum by FFT synthesis over the full cadence grid, sampled at the
cadences in use, so gaps do not change the spectrum.

## Weight models

A {class}`~systematix.generated.WeightModel` holds one (quarter, channel)'s donors:

| Attribute | Shape | Meaning |
|---|---|---|
| `keys` | (k,) | weight keys (see above) |
| `kic`, `kepmag` | (n,) | donor KIC ids and Kepler magnitudes |
| `W`, `Ve` | (n, k), (n, k, k) | fitted weights and their fit covariances |
| `jitter`, `white_jitter` | (n,) | fitted jitter and its white part |
| `red`, `red_edges` | (n, nbins), (nbins+1,) | pre-fit red spectrum per donor, and its frequency bins (c/d) |
| `steps`, `t0c` | | the output's consensus steps and canonical ramp start times |
| `mean`, `cov` | (k,), (k, k) | moment model of the weights |

`cov` is the donors' sample covariance minus their mean fit covariance, projected to
positive semi-definite; `clipped` is the share of the trace removed by that
projection. Keys present in fewer than 90 % of the donors are dropped
({meth}`WeightModel.estimate <systematix.generated.WeightModel.estimate>`).

```python
m = sx.load_weight_model("/path/to/models", quarter=2, channel=65)
m.keys[:3], m.W.shape, m.sd      # sd: square root of diag(cov)
m.subset(kics)                   # re-estimate on a subset of donors (cross-validation)
```

## Drawing realizations

{class}`~systematix.generated.Generated` needs a template
{class}`~systematix.observation.Observation` with `MODULE`, `OUTPUT` and `CHANNEL`
in each quarter's `quarter_meta` (for the CBVs and the model files) and `KEPMAG`
(for magnitude matching).
{meth}`Observation.from_systematics <systematix.observation.Observation.from_systematics>`
provides all of these. The template supplies the cadence, gaps, quality flags,
crowding, flux errors and per-quarter flux level.

`draw="resample"` (default)
: Take one donor's fitted weights in every quarter, and that donor's noise. A donor
  with models in all the template's quarters is used when there is one, otherwise
  one donor per quarter. Whole donors are resampled because the weights are
  heavy-tailed: in the Phase 4 pilot a Gaussian with the right covariance matched
  the mean power but was 1.3–1.4× too strong at the median.

`draw="gaussian"`
: Draw weights from the moment model (`mean`, `cov`), with a random donor's noise.

`kp_match="auto"`
: With resampling, draw from the 50 donors closest in Kepler magnitude to the
  template ({data}`~systematix.generated.KP_MATCH`). Pass an integer for a different
  number, or `None` to pool donors of every magnitude. Matching is the default
  because the relative systematics grow with Kp. In the full Phase 4 gate (1,460
  output-quarters), donors pooled over Kp 12–15 gave S3 a median power ratio of
  0.56 on templates brighter than Kp 14 and 1.38 on fainter ones; the 50 nearest
  donors gave 0.91 and 0.99, and passed in 98.8 % and 99.9 % of the cells.

`red_noise=True`
: Add red noise with the donor's residual spectrum and use the white part of its
  jitter, so the total noise variance is unchanged and only its spectrum moves.
  The red noise is part of the realization's noise, not its `scale`: each
  {meth}`Systematics.draw_noise <systematix.sources.Systematics.draw_noise>` call
  draws it afresh. It needs models with residual spectra; pass `red_noise=False`
  for older models.

`blocks`, `jitter`
: Switch design blocks (`"offsets"`, `"polynomial"`, `"ramps"`, `"cbv"`) and the
  extra white noise on or off, to test their effects one at a time.

`design`
: Precomputed {func}`~systematix.decompose.design_blocks` output for the template,
  to skip rebuilding it (and loading CBVs) when making several `Generated` objects on
  one template.

Ramps at gaps in the template that the donors do not share get independent draws at
the model's pooled ramp standard deviation.

Besides {meth}`~systematix.generated.Generated.realize`,
{meth}`~systematix.generated.Generated.draw_trend` returns the relative systematics
$s(t)$, the white jitter per quarter and the donor, and
{meth}`~systematix.generated.Generated.draw_weights` returns the weights alone.
