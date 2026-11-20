"""Sphinx configuration for the systematix documentation."""

from __future__ import annotations

import os
import sys
import warnings
from datetime import date

# Document the source tree, so a local build needs no install (src layout).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import systematix

# -- Project ---------------------------------------------------------------

project = "systematix"
author = "Jessica Birky"
copyright = f"{date.today().year}, {author}"
release = systematix.__version__
version = ".".join(release.split(".")[:2])

# -- General ---------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Docstrings use ``literal`` for code; make single backticks code too.
default_role = "code"

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- autodoc / napoleon ------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_preserve_defaults = True
autosummary_generate = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "array": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
    "Table": "astropy.table.Table",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "lightkurve": ("https://lightkurve.github.io/lightkurve/", None),
}

# -- HTML ------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = f"systematix {release}"
html_static_path = ["_static"]
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
    # Add once the repository is public:
    # "repository_url": "https://github.com/<user>/systematix",
    # "use_repository_button": True,
    # "use_issues_button": True,
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def _drop_type_docstrings(app, what, name, obj, options, lines):
    """Module constants would otherwise show their type's docstring (dict, tuple, ndarray, ...).

    None of systematix's constants has a ``#:`` doc comment; their descriptions are
    written in the API pages instead.
    """
    import numpy as np

    if what == "data" and isinstance(obj, (dict, tuple, list, np.ndarray, int, float, str)):
        lines.clear()


def setup(app):
    app.connect("autodoc-process-docstring", _drop_type_docstrings)
