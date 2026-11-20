# Installation

systematix needs Python 3.10 or later. Install it from a clone of the repository:

```bash
pip install -e .            # from the repository root
pip install -e ".[test]"    # with pytest
```

## Dependencies

| Package | Used for |
|---|---|
| `numpy`, `astropy` | arrays, tables, units, FITS |
| `lightkurve` ≥ 2.4 | searching and downloading Kepler light curves from MAST |
| `scipy` | ARC2 regression and red-noise synthesis (installed with lightkurve) |
| `h5py` | reading and writing S3 weight models |
| `matplotlib` | plots (`--plot`, {func}`~systematix.transfer.plot_transfer`); optional |

```{note}
`import systematix` imports `h5py`, which `pyproject.toml` does not list yet.
If the import fails with `ModuleNotFoundError: No module named 'h5py'`, run
`pip install h5py`.
```

## Downloaded data and caches

systematix downloads three kinds of data. Each is fetched once and cached.

| Data | Size | Cache |
|---|---|---|
| Kepler light curves (via lightkurve) | ~0.5 MB per quarter | `~/.lightkurve/cache`, or `download_dir=` / `--download-dir` |
| DR25 cotrending basis vectors (CBVs) | ~25 MB per quarter, ~450 MB for all 18 | `$SYSTEMATIX_CACHE/cbv` |
| McQuillan et al. (2014) non-periodic table | ~5 MB | `$SYSTEMATIX_CACHE/donors` |

`$SYSTEMATIX_CACHE` defaults to `~/.systematix`. Set it to put the CBVs and donor
table somewhere else, for example a scratch disk on a cluster:

```bash
export SYSTEMATIX_CACHE=/scratch/$USER/systematix
```

Downloads are written atomically, so an interrupted download leaves no partial file.

## Optional: the original ARC2 code

{class}`~systematix.detrend.ARC2` has a native MIT reimplementation, used by
default. To compare against the original GPL-3 code,
[OxKeplerSC](https://github.com/OxES/OxKeplerSC), clone it and point systematix
at the clone. It cannot be pip-installed on current Python, because its
`setup.py` needs `numpy.distutils`.

```bash
git clone https://github.com/OxES/OxKeplerSC.git
export OXKEPLERSC_PATH=$PWD/OxKeplerSC
```

Then use `sx.ARC2(backend="oxksc")`. See {ref}`arc2` for details. Nothing from
OxKeplerSC is distributed with systematix.

## Running the tests

```bash
pytest              # offline, synthetic data only
pytest -m remote    # also downloads Kepler-10, CBVs and the donor table
```

With `OXKEPLERSC_PATH` set, `pytest` also checks the native ARC2 against
OxKeplerSC, and against the corrected light curves stored in its repository.

## Building these docs

```bash
pip install -r docs/requirements.txt
make -C docs html          # output in docs/_build/html
```
