# Injection: sources and the observation model

An injection-recovery test needs a realistic instrument: the real cadence and gaps,
per-quarter flux levels, dilution by neighbouring stars, systematics and noise.
systematix splits this into three pieces:

- a **source** produces a {class}`~systematix.sources.Systematics` realization on a
  real (or synthetic) cadence;
- {func}`~systematix.sources.inject` combines it with your stellar signal and
  returns an {class}`~systematix.observation.Observation`;
- a **detrender** ({doc}`detrenders`) processes the observation.

```python
import numpy as np
import systematix as sx

donor = sx.quiet_donors(teff=(5500, 6000), logg_min=4.2)[0]
sys = sx.PDCRatio(donor, quarters=[2, 3, 4, 5]).realize()

f_spot = 1 + 1e-3 * np.sin(2 * np.pi * sys.time / 12.0)   # your model, normalized to 1
obs = sx.inject(sys, f_spot, rng=0)
```

## The observation model

{func}`~systematix.sources.inject` computes

$$
F_\mathrm{obs}(t) = L_q\, m_q(t)\,\big[\,c_q\, F_\mathrm{spot}(t) + (1 - c_q)\,\big] + n(t),
$$

| Symbol | Meaning | In the code |
|---|---|---|
| $F_\mathrm{spot}(t)$ | stellar signal, normalized to 1 | `f_spot` |
| $L_q$ | flux level of quarter $q$ (e⁻/s) | part of `sys.scale` |
| $m_q(t)$ | multiplicative systematics | part of `sys.scale` |
| $c_q$ | `CROWDSAP` of quarter $q$: dilution by neighbouring stars | `sys.crowding()` |
| $n(t)$ | white noise of standard deviation `sys.sigma`, plus red noise if the source has any | `sys.draw_noise(rng)` |

The product $L_q m_q(t)$ is stored as one array, `Systematics.scale`. Dilution can
be switched off per source with `dilution=False`.

The returned observation's `truth` dictionary records what went in: `f_spot`,
`scale`, `crowding`, `noise`, `source` and `donor`.

### Paired trials

Pass the same `noise` array to compare runs with and without a signal. Systematics
and noise then cancel in the difference, which is how the transfer functions
({doc}`transfer`) work:

```python
noise = sys.draw_noise(rng=1)
obs_null = sx.inject(sys, 1.0, noise=noise)
obs_sig = sx.inject(sys, f_spot, noise=noise)
```

## Sources

Every source has a `realize(rng=None)` method that returns a
{class}`~systematix.sources.Systematics`.

| ID | Source | `scale` | Noise added | Use it for |
|---|---|---|---|---|
| S0 | {class}`~systematix.sources.NoSystematics` | 1 (or each quarter's median, with `offsets=True`) | the template's relative errors, or `sigma` | the cadence and gaps alone; a baseline |
| S1 | {class}`~systematix.sources.PDCRatio` | $L_q \times$ the donor's `ratio` = SAP / PDC | white, at the donor's `sap_flux_err` | the systematics PDC removed from a real star |
| S2 | {class}`~systematix.sources.QuietSAP` | the donor's SAP flux itself | none (the SAP already has real noise) | real noise, granulation and systematics PDC cannot model |
| S3 | {class}`~systematix.generated.Generated` | $L_q (1 + M w)$, drawn from donor weights | white plus jitter, plus red noise | new systematics realizations on any star's cadence; see {doc}`generated` |
| | {class}`~systematix.sources.DonorPool` | S1 or S2 from a random donor | as S1 or S2 | averaging over donors |

S1 and S2 take a donor: a KIC id (downloaded with
{func}`~systematix.core.get_systematics`) or a light curve that function already
returned. A donor is loaded once and cached on the source, so later `realize()`
calls return the same realization; only the noise drawn by `inject` changes.

```{admonition} S1 and CBV detrenders
:class: warning
PDC builds its correction from CBVs, so a CBV detrender recovers S1 systematics
almost perfectly. S1 flatters CBV-based detrenders. S2 or S3 are fairer tests.
```

```{admonition} Donors should be quiet
:class: tip
For S1, the ratio carries any long-period stellar signal PDC removed. For S2, the
SAP flux carries all of the donor's variability. Use quiet donors in both cases.
```

### Quiet donors

{func}`~systematix.donors.quiet_donors` returns KIC ids from Table 2 of McQuillan,
Mazeh & Aigrain (2014, ApJS 211, 24): the 99,000 main-sequence Kepler targets
without a reliable rotation period. By default it keeps the 68,246 with no candidate
period at all. "Quiet" here means no detected period, not no variability.

```python
kics = sx.quiet_donors(teff=(5500, 6000), logg_min=4.2)   # Sun-like, dwarfs
table = sx.load_nonperiodic()                             # the full table
```

### Donor pools

{class}`~systematix.sources.DonorPool` draws a new donor for each realization:

```python
pool = sx.DonorPool(kics, kind="quiet_sap", quarters=[2, 3, 4, 5])
sys = pool.realize(rng=42)
sys.donor                   # which star was used
```

`kind` is `"pdc_ratio"` (S1, the default) or `"quiet_sap"` (S2). The optional
`max_pdcvar` rejects donors whose mean `PDCVAR` (PDC's variability estimate, from
the FITS headers) exceeds it. Donors with no data in the requested quarters are
skipped. Up to `max_tries` donors are tried before a `RuntimeError`.

## Working offline

{func}`~systematix.sources.synthetic_cadence` builds a Kepler-like long-cadence
template with monthly downlink gaps and flux 1, so S0 and tests run with no
downloads:

```python
template = sx.synthetic_cadence(n_quarters=2, sigma=1e-4)
sys = sx.NoSystematics(template).realize()
obs = sx.inject(sys, f_spot(template.time), rng=0)
```

## The Observation container

{class}`~systematix.observation.Observation` is what sources produce and
detrenders consume: `time`, `flux`, `flux_err`, `quarter`, `quality`, `cadenceno`,
`quarter_meta` (each quarter's header values), `kic` and `truth`. `flux` is in
SAP-like units with a different level per quarter; detrenders normalize it
themselves.

```python
obs = sx.Observation.from_systematics(sx.get_systematics(kic))   # SAP flux of a real star
obs.quarters                    # [2, 3, 4, ...]
obs.crowding()                  # CROWDSAP per cadence
obs.segment_starts(min_gap=0.5) # indices where data resume after a gap
obs.select(obs.quality == 0)    # subset of cadences (truth arrays are subset too)
```

## Binning

{func}`systematix.sources.bin_systematics` averages a realization onto bins of `dt`
days within each quarter, with inverse-variance weights from `flux_err`. White noise
averages down accordingly, and red noise is binned the same way, so injecting into
the binned realization is equivalent to injecting at the bin centres and then
binning. {meth}`~systematix.detrend.DetrendResult.binned` does the same for a
detrended light curve.
