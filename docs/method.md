# What "SAP − PDCSAP" means here

Kepler light-curve files carry two fluxes. `SAP_FLUX` is simple aperture
photometry: the sum of the pixels in the optimal aperture, after background
subtraction. `PDCSAP_FLUX` is the same flux after the Presearch Data Conditioning
(PDC) module has processed it. systematix defines the instrumental systematics as
what PDC removed. A literal difference of the two columns does not give that,
because PDC does two separate things.

## The two steps of PDC

**1. Cotrending.** PDC's MAP algorithm (multi-scale MAP, msMAP) fits a systematics
model built from cotrending basis vectors and subtracts it. Call the result $F_\mathrm{cot}(t)$. It is
still on the SAP scale. The systematics we want are

$$
s(t) = \mathrm{SAP}(t) - F_\mathrm{cot}(t).
$$

**2. Crowding and flux-fraction correction** (Kinemuchi et al. 2012, PASP 124, 963).
PDC then removes the flux from neighbouring stars and rescales for the target flux
that falls outside the aperture:

$$
\mathrm{PDCSAP}(t) = \frac{F_\mathrm{cot}(t) - (1 - c)\,\widetilde{F}_\mathrm{cot}}{f},
$$

where $c$ is the file's `CROWDSAP` (the fraction of flux in the aperture that comes
from the target), $f$ is `FLFRCSAP` (the fraction of the target's flux that falls in
the aperture), and $\widetilde{F}$ is a median over the quarter.

## Why a literal difference is wrong

Subtracting the two archived columns mixes step 2 into the result:

$$
\mathrm{SAP} - \mathrm{PDCSAP} = s(t) + \Big(1 - \frac{1}{f}\Big)\,F_\mathrm{cot}(t)
+ \frac{1 - c}{f}\,\widetilde{F}_\mathrm{cot}.
$$

- The constant terms add a large offset: about −10,000 e⁻/s for Kepler-10 in Q3.
- The $(1 - 1/f)\,F_\mathrm{cot}(t)$ term leaks a fraction $1 - 1/f$ of the star's
  own variability into the "systematics".

## What systematix does

By default ({func}`~systematix.core.compute_systematics` with
`undo_crowding=True`) systematix inverts step 2 with each file's `CROWDSAP` and
`FLFRCSAP` header values. Taking the median of the PDCSAP equation gives
$\widetilde{\mathrm{PDCSAP}} = c\,\widetilde{F}_\mathrm{cot}/f$, so

$$
F_\mathrm{cot}(t) = f\,\mathrm{PDCSAP}(t) + \frac{(1 - c)\,f}{c}\,\widetilde{\mathrm{PDCSAP}}.
$$

This is the `pdcsap_flux_rescaled` column. The systematics are then

$$
s(t) = \mathrm{SAP}(t) - F_\mathrm{cot}(t),
$$

which holds only the cotrending correction. The `ratio` column,
$\mathrm{SAP}(t) / F_\mathrm{cot}(t)$, is the same correction written as a
multiplicative factor; source S1 ({class}`~systematix.sources.PDCRatio`) uses it.

Pass `undo_crowding=False` (CLI: `--literal`) to get the plain column difference
instead.

The medians are taken over the cadences that are finite in both SAP and PDCSAP
after quality masking, which should match the cadences PDC used. If they differ
slightly, a small constant offset per quarter remains; the time-varying shape is
unaffected.

## Normalization

With `normalize=True`, each file's $s(t)$ is divided by that file's median SAP
flux. The result is the fractional systematics of that quarter, ready to add to a
simulated light curve with median 1.
