"""Command line: ``systematix 11904151 -q 2 3 4 --normalize -o kepler10_sys.csv``."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from astropy.table import Table


def to_table(lc) -> Table:
    """Flat table with time as float BKJD and the systematics in a ``systematics`` column."""
    t = Table()
    t["time"] = np.asarray(lc.time.value, dtype=float)
    t["time"].unit = "d"
    t["time"].description = "BJD - 2454833 (BKJD)"
    t["systematics"] = lc.flux
    for col in ("sap_flux", "sap_flux_err", "pdcsap_flux", "pdcsap_flux_rescaled", "ratio", "quality", "cadenceno", "quarter"):
        t[col] = lc[col]
    # Keep only header-friendly scalars so the table also writes cleanly to FITS.
    t.meta = {k: v for k, v in lc.meta.items() if isinstance(v, (str, int, float, bool, np.number))}
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="systematix",
        description="Download Kepler light curves for a KIC target and write SAP - PDCSAP systematics.",
    )
    p.add_argument("kic", help="KIC id, e.g. 11904151 or 'KIC 11904151'")
    p.add_argument("-q", "--quarters", type=int, nargs="+", help="quarters to include (default: all)")
    p.add_argument("-c", "--cadence", choices=["long", "short"], default="long")
    p.add_argument("-n", "--normalize", action="store_true", help="divide by each quarter's median SAP flux")
    p.add_argument(
        "--literal",
        action="store_true",
        help="literal SAP - PDCSAP, without undoing PDC's crowding/flux-fraction correction",
    )
    p.add_argument("--quality-bitmask", default="default", help="lightkurve quality bitmask (default: 'default')")
    p.add_argument("--keep-nan", action="store_true", help="keep cadences where SAP or PDCSAP is NaN")
    p.add_argument("--download-dir", help="lightkurve cache directory")
    p.add_argument(
        "-o", "--output",
        help="output file; format from extension (.csv, .ecsv, .fits). Default: kic<ID>_systematics.csv",
    )
    p.add_argument("--plot", metavar="PNG", help="also save a plot of SAP, PDCSAP and the systematics")
    args = p.parse_args(argv)

    from .core import get_systematics, parse_kic

    bitmask = int(args.quality_bitmask) if args.quality_bitmask.isdigit() else args.quality_bitmask
    try:
        lc = get_systematics(
            args.kic,
            quarters=args.quarters,
            cadence=args.cadence,
            normalize=args.normalize,
            undo_crowding=not args.literal,
            quality_bitmask=bitmask,
            dropna=not args.keep_nan,
            download_dir=args.download_dir,
        )
    except (LookupError, ValueError) as e:
        print(f"systematix: {e}", file=sys.stderr)
        return 1

    out = args.output or f"kic{parse_kic(args.kic)}_systematics.csv"
    to_table(lc).write(out, overwrite=True)
    print(f"wrote {len(lc)} cadences from quarters {lc.meta['QUARTERS']} to {out}")

    if args.plot:
        _plot(lc, args.plot)
        print(f"wrote {args.plot}")
    return 0


def _plot(lc, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = lc.time.value
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, lc["sap_flux"].value, ".", ms=1, color="0.3")
    axes[0].set_ylabel("SAP [e-/s]")
    axes[1].plot(t, lc["pdcsap_flux_rescaled"].value, ".", ms=1, color="C0")
    axes[1].set_ylabel("PDCSAP\n(SAP scale) [e-/s]")
    axes[2].plot(t, lc.flux.value, ".", ms=1, color="C3")
    axes[2].set_ylabel("systematics" + ("" if lc.meta["NORMALIZED"] else " [e-/s]"))
    axes[2].set_xlabel("Time [BKJD]")
    axes[0].set_title(lc.meta["LABEL"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
