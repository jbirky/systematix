"""Optional access to the original ARC2 code, OxKeplerSC (GPL-3).

OxKeplerSC (https://github.com/OxES/OxKeplerSC) cannot be pip-installed on
current Python: its ``setup.py`` needs ``numpy.distutils``. So it is found
at runtime, in this order:

1. an importable ``oxksc`` package (if you built it yourself), or
2. a clone of the repository at ``$OXKEPLERSC_PATH``.

It calls ``scipy.log`` and ``scipy.multiply``, which modern scipy removed;
:func:`scipy_aliases` adds them back for the duration of a call only.
Nothing from OxKeplerSC is distributed with systematix.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

ENV_VAR = "OXKEPLERSC_PATH"
_HELP = (
    "OxKeplerSC not found. Clone https://github.com/OxES/OxKeplerSC and set "
    f"{ENV_VAR}=/path/to/OxKeplerSC (it cannot be pip-installed on current Python)."
)


def load_oxksc_cbv(path: str | os.PathLike | None = None):
    """Return OxKeplerSC's ``oxksc.cbvc.cbv`` module, or raise ImportError."""
    try:
        if path is None:
            return importlib.import_module("oxksc.cbvc.cbv")
    except ImportError:
        pass
    root = path or os.environ.get(ENV_VAR)
    if not root:
        raise ImportError(_HELP)
    root = Path(root).expanduser()
    src = root / "src" if (root / "src" / "cbvc").is_dir() else root
    if not (src / "cbvc" / "cbv.py").is_file():
        raise ImportError(f"{src} does not look like an OxKeplerSC checkout. {_HELP}")
    if "oxksc" not in sys.modules or Path(sys.modules["oxksc"].__path__[0]) != src:
        for name in [m for m in sys.modules if m == "oxksc" or m.startswith("oxksc.")]:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            "oxksc", src / "__init__.py", submodule_search_locations=[str(src)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["oxksc"] = module
        spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("oxksc.cbvc.cbv")


def available() -> bool:
    try:
        load_oxksc_cbv()
        return True
    except ImportError:
        return False


@contextmanager
def scipy_aliases():
    """Temporarily restore ``scipy.log`` and ``scipy.multiply`` (removed in scipy 1.12+)."""
    import numpy as np
    import scipy

    added = []
    for name in ("log", "multiply"):
        if not hasattr(scipy, name):
            setattr(scipy, name, getattr(np, name))
            added.append(name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # np.matrix PendingDeprecationWarning
            yield
    finally:
        for name in added:
            delattr(scipy, name)
