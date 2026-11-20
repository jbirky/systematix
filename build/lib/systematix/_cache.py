"""Download cache shared by the donor-list and CBV loaders."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

ENV_VAR = "SYSTEMATIX_CACHE"


def cache_dir(subdir: str = "", base: str | os.PathLike | None = None) -> Path:
    """``$SYSTEMATIX_CACHE`` (default ``~/.systematix``), plus ``subdir``."""
    root = Path(base or os.environ.get(ENV_VAR) or Path.home() / ".systematix")
    path = root / subdir if subdir else root
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path, timeout: float = 300.0) -> Path:
    """Fetch ``url`` to ``dest`` unless it already exists. Writes atomically."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(url, timeout=timeout) as resp:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest
