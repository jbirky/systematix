# Command line

Installing the package adds a `systematix` command. It downloads a target's light
curves, computes the systematics and writes them to a table.

```bash
systematix 11904151                                  # -> kic11904151_systematics.csv
systematix "KIC 11904151" -q 2 3 4 -n -o sys.ecsv    # quarters 2-4, normalized
systematix 11904151 --plot sys.png                   # plus a diagnostic plot
systematix 11904151 --cadence short                  # 1-min data, where available
```

`python -m systematix` does the same.

## Options

| Option | Meaning |
|---|---|
| `kic` | KIC id, e.g. `11904151` or `'KIC 11904151'` |
| `-q`, `--quarters Q [Q ...]` | quarters to include (default: all) |
| `-c`, `--cadence {long,short}` | 30-min (default) or 1-min cadence |
| `-n`, `--normalize` | divide each quarter by its median SAP flux |
| `--literal` | literal SAP − PDCSAP, without undoing PDC's crowding and flux-fraction correction |
| `--quality-bitmask MASK` | lightkurve quality bitmask: `none`, `default` (default), `hard`, `hardest` or an integer |
| `--keep-nan` | keep cadences where SAP or PDCSAP is NaN |
| `--download-dir DIR` | lightkurve cache directory |
| `-o`, `--output FILE` | output file (default `kic<ID>_systematics.csv`) |
| `--plot PNG` | also save a plot of SAP, PDCSAP and the systematics |

The output format comes from the file extension: `.csv`, `.ecsv` or `.fits`.

## Output table

| Column | Unit | Meaning |
|---|---|---|
| `time` | d | BKJD (BJD − 2454833) |
| `systematics` | e-/s, or none with `-n` | SAP minus PDCSAP on the SAP scale |
| `sap_flux`, `sap_flux_err` | e-/s | raw SAP flux |
| `pdcsap_flux` | e-/s | PDCSAP flux as archived |
| `pdcsap_flux_rescaled` | e-/s | PDCSAP on the SAP scale |
| `ratio` | | `sap_flux / pdcsap_flux_rescaled` |
| `quality`, `cadenceno`, `quarter` | | from the FITS files |

The table header keeps the scalar metadata (`KEPLERID`, `CADENCE`, `NORMALIZED`,
...). The per-quarter dictionaries (`CROWDSAP`, `QUARTER_META`) are left out so
the table also writes to FITS. Use the Python API if you need them.

The command exits with status 1 and a one-line message if the target has no
Kepler light curves or the arguments are invalid.
