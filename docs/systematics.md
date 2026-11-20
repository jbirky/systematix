# Kepler systematics for a KIC target

{func}`~systematix.core.get_systematics` downloads every Kepler light curve file
for a target, computes the systematics in each file separately, and stitches them
into one {class}`lightkurve.LightCurve`.

```python
import systematix as sx

lc = sx.get_systematics("KIC 11904151")                 # all quarters, e-/s
lc = sx.get_systematics(11904151, quarters=[2, 3, 4])   # a subset
lc = sx.get_systematics(11904151, normalize=True)       # fractional, per quarter
lc = sx.get_systematics(11904151, cadence="short")      # 1-min data, where it exists

lc.flux                     # the systematics
lc["sap_flux"], lc["pdcsap_flux"], lc["quarter"]
lc.plot()
lc.to_pandas()
```

The KIC id can be an integer or any string with exactly one number in it:
`11904151`, `"KIC 11904151"` and `"kplr011904151"` all work
({func}`~systematix.core.parse_kic`).

## Options

`quarters`
: An int or list of ints from 0 to 17. Default: every quarter on MAST.

`cadence`
: `"long"` (30 min, the default) or `"short"` (1 min). Short cadence has three
  files per quarter, one per month; each is treated as its own segment.

`normalize`
: If False (default), the systematics are in e-/s. If True, each file's
  systematics are divided by that file's median SAP flux. The result is a
  dimensionless fraction you can add to a simulated light curve normalized to 1.

`undo_crowding`
: If True (default), undo PDC's crowding and flux-fraction correction before
  differencing. Set False for the literal `SAP_FLUX - PDCSAP_FLUX`. See
  {doc}`method`.

`quality_bitmask`
: Passed to lightkurve: `"none"`, `"default"`, `"hard"`, `"hardest"` or an int.

`dropna`
: Drop cadences where SAP or PDCSAP is NaN (default True).

`download_dir`
: lightkurve cache directory (default `~/.lightkurve/cache`).

## What comes back

`flux` holds the systematics. The other columns are:

| Column | Meaning |
|---|---|
| `flux` | systematics: `sap_flux - pdcsap_flux_rescaled` (e-/s, or dimensionless if `normalize=True`) |
| `flux_err` | NaN on purpose; see below |
| `sap_flux`, `sap_flux_err` | raw SAP flux (e-/s) |
| `pdcsap_flux` | PDCSAP flux as archived (e-/s) |
| `pdcsap_flux_rescaled` | PDCSAP put back on the SAP scale (e-/s); see {doc}`method` |
| `ratio` | multiplicative systematics, `sap_flux / pdcsap_flux_rescaled` |
| `quality`, `cadenceno`, `quarter` | from the FITS files |

`flux_err` is NaN because the pipeline gives no independent uncertainty for the
difference of two fluxes built from the same photons. For realistic photon noise
use `sap_flux_err`.

`sap_flux`, `pdcsap_flux` and `pdcsap_flux_rescaled` are always in e-/s, even when
`normalize=True`.

`lc.meta` holds:

| Key | Value |
|---|---|
| `KEPLERID`, `OBJECT`, `LABEL` | the target |
| `CADENCE`, `NORMALIZED`, `UNDO_CROWDING`, `QUALITY_BITMASK` | the options used |
| `QUARTERS` | sorted list of quarters returned |
| `CROWDSAP`, `FLFRCSAP` | `{quarter: value}` (a list of three values per quarter for short cadence) |
| `QUARTER_META` | `{quarter: {key: value}}` for every key in {data}`~systematix.core.HEADER_KEYS` |

`HEADER_KEYS` are the per-file FITS header values that the injection-recovery
tools use: `CROWDSAP`, `FLFRCSAP`, `MODULE`, `OUTPUT`, `CHANNEL`, `SEASON`,
`KEPMAG`, `PDCVAR`, `CDPP3_0`, `CDPP6_0`, `CDPP12_0`. `MODULE` and `OUTPUT` pick
the CBVs for each quarter, `CHANNEL` picks the S3 weight model, and `KEPMAG`
matches donors by brightness.

## Adding systematics to a simulation

The simplest use is to add one star's normalized systematics to a simulated light
curve that is normalized to 1:

```python
sys = sx.get_systematics(kic, normalize=True)
f_obs = f_sim(sys.time.value) + sys.flux.value
```

`time` is BKJD (BJD − 2454833). For a fuller model, with per-quarter flux levels,
dilution by neighbouring stars, noise and a choice of systematics sources, see
{doc}`injection`.

## Lower-level functions

{func}`~systematix.core.get_systematics` is built from three functions you can
call yourself:

{func}`~systematix.core.download_lightcurves`
: Searches MAST for the target's pipeline light curves and downloads them.
  `search_lightcurve` matches by sky position, so nearby stars that come back
  are dropped. Raises `LookupError` if nothing is found.

{func}`~systematix.core.systematics_from_lightcurve`
: The systematics for one file already loaded by lightkurve.

{func}`~systematix.core.compute_systematics`
: The arithmetic on plain arrays: `(systematics, pdcsap_rescaled)` from SAP,
  PDCSAP, `CROWDSAP` and `FLFRCSAP`.

```python
import lightkurve as lk

lc = lk.read("kplr011904151-2009350155506_llc.fits")
sys_q3 = sx.systematics_from_lightcurve(lc, normalize=True)
```

## Caveats

- Each quarter uses a different aperture on a different CCD, so the systematics
  are computed per file and are **not continuous across quarter boundaries**.
- **PDC-MAP partly removes real astrophysical signal on long timescales**
  (≳ 20 days, for example slow rotators). That signal shows up in the
  "systematics" here, because they are defined as whatever PDC removed. Use quiet
  stars ({func}`~systematix.donors.quiet_donors`) when you want systematics
  alone.
- Undoing crowding uses `median(PDCSAP)` over the cadences that survive quality
  masking. If those differ slightly from the cadences PDC used, a small constant
  offset per quarter remains. The time-varying shape is not affected.
