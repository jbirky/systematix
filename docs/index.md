# systematix

**Instrumental systematics for light-curve injection-recovery tests.** Kepler is the
mission implemented so far.

systematix does two jobs:

1. **Get the systematics of any KIC target.** It downloads the official Kepler
   light curves from MAST and returns the raw SAP flux minus the PDC-corrected
   PDCSAP flux, one quarter at a time, stitched into one light curve. By default
   it first undoes PDC's crowding and flux-fraction correction, so the result holds
   only the cotrending correction PDC applied.
2. **Supply the instrument side of an injection-recovery test.** It provides
   systematics *sources* (S0 to S3), an *injection model*, *detrenders* with a
   common interface, and *transfer functions* that measure how much of an injected
   signal each detrender keeps.

The package does not depend on any stellar model. spotgp, butterpy or any other
forward model can supply the signal you inject.

```python
import systematix as sx

lc = sx.get_systematics("KIC 11904151", quarters=[2, 3, 4], normalize=True)
lc.plot()                                   # a lightkurve.LightCurve

sys = sx.PDCRatio(11904151, quarters=[2, 3, 4]).realize()
obs = sx.inject(sys, f_spot, rng=0)         # f_spot: your model, normalized to 1
res = sx.CBV(n_cbv=8)(obs)                  # detrend
```

## Where to start

| If you want to... | Read |
|---|---|
| install the package and set up caches | {doc}`install` |
| get the systematics of a star in Python | {doc}`systematics` |
| do the same from the shell | {doc}`cli` |
| understand what "SAP − PDCSAP" means here | {doc}`method` |
| inject a signal into realistic Kepler systematics | {doc}`injection` |
| detrend the result, or marginalize over a nuisance model | {doc}`detrenders` |
| measure which periods survive detrending | {doc}`transfer` |
| draw new systematics from donor-star weights (S3) | {doc}`generated` |
| look up a function | {doc}`api/index` |

```{toctree}
:hidden:
:caption: Getting started

install
systematics
cli
method
```

```{toctree}
:hidden:
:caption: Injection-recovery

injection
detrenders
transfer
generated
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
