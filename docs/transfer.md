# Transfer functions

Every detrender removes some real signal along with the systematics. Long periods
suffer most: a signal much longer than a quarter is largely absorbed by the
per-quarter normalization alone. {mod}`systematix.transfer` measures how much of an
injected sinusoid survives each detrender, as a function of period, without fitting
any stellar model.

```python
import numpy as np
import systematix as sx

sys = sx.PDCRatio(donor, quarters=[2, 3, 4, 5]).realize()
tab = sx.transfer_sweep(
    [sx.QuarterMedian(), sx.CBV(8), sx.RollingMedian(10)],
    sys,
    periods=np.geomspace(0.5, 400, 28),
)
sx.plot_transfer(tab)
```

## Method

Each trial runs the detrender twice on the same systematics and the same noise:
once with the signal $f(t)$ injected and once without. The difference of the two
outputs is the *recovered* signal. Systematics and noise cancel in it, exactly for
linear detrenders and approximately for nonlinear ones.

The recovered signal is compared with the *injected* signal as it appears in
relative flux,

$$
\mathrm{inj}(t) = c_q\,\big(f(t) - 1\big)\,\frac{\mathrm{scale}(t)}{\widetilde{\mathrm{scale}}_q},
$$

which accounts for dilution and for the multiplicative systematics. Two numbers
summarize the comparison:

gain
: $\langle \mathrm{rec}, \mathrm{inj}\rangle / \langle \mathrm{inj}, \mathrm{inj}\rangle$,
  the in-phase amplitude retained. 1 means the signal is kept, 0 means it is
  removed. The power retained is $\mathrm{gain}^2$.

distortion
: $1 - \langle \mathrm{rec}, \mathrm{inj}\rangle^2 / \big(\langle \mathrm{rec}, \mathrm{rec}\rangle\langle \mathrm{inj}, \mathrm{inj}\rangle\big)$,
  the share of the recovered signal that is not a scaled copy of the injected one.

## `transfer_sweep`

{func}`~systematix.transfer.transfer_sweep` injects `n_phase` sinusoids
$1 + A \sin(2\pi t/P + \phi)$ at each period $P$, with evenly spaced phases and a
random offset. Every detrender sees the same noise and the same signals. Detrenders
can be objects or config dictionaries (see {func}`~systematix.detrend.make_detrender`).

It returns an {class}`astropy.table.Table` with one row per detrender and period:

| Column | Meaning |
|---|---|
| `detrender` | the detrender's label, e.g. `cbv(n_cbv=8, order=0)` |
| `period` | injected period (d) |
| `gain`, `gain_std` | mean and standard deviation of the gain over phases |
| `power` | mean of gain² |
| `distortion` | mean distortion |

{func}`~systematix.transfer.plot_transfer` draws power retained against period on
log axes, one line per detrender, with a dashed line at 50 %.

For a single signal, {func}`~systematix.transfer.signal_gain` returns
`(gain, distortion)`. Pass the null run as `baseline` to reuse it across signals:

```python
noise = sys.draw_noise(rng=0)
d = sx.CBV(8)
base = d(sx.inject(sys, 1.0, noise))
gain, distortion = sx.signal_gain(d, sys, f, noise, baseline=base)
```

## Example script

`examples/transfer_sweep.py` runs the full sweep. It draws a random quiet Sun-like
donor (5500–6000 K, log g ≥ 4.2), uses its S1 systematics on quarters 2–5, and
compares eight detrenders from 0.5 to 400 days:

```bash
python examples/transfer_sweep.py --outdir transfer_out
python examples/transfer_sweep.py --donor 11904151 --quarters 2 3 4 5 --n-phase 8
```

It writes `transfer.ecsv` and `transfer.png`, and prints the longest period at which
each detrender still keeps half the power. Options: `--donor`, `--quarters`,
`--n-periods` (28), `--n-phase` (4), `--amplitude` (1e-3), `--seed` (0) and
`--outdir`.

```{note}
The example uses S1, which flatters CBV detrenders because PDC built S1 from CBVs.
Swap in {class}`~systematix.sources.QuietSAP` or
{class}`~systematix.generated.Generated` for a fairer comparison.
```
