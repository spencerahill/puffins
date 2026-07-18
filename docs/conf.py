"""Sphinx configuration for the puffins documentation."""

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "puffins"
author = "Spencer A. Hill"
copyright = "2026, Spencer A. Hill"  # noqa: A001

try:
    release = importlib.metadata.version("puffins")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Roadmaps are internal planning docs written in Markdown; they are not part
# of the rendered site and Sphinx should not try to parse them.
exclude_patterns = ["_build", "roadmaps", "Thumbs.db", ".DS_Store"]

# NumPy-style ``References`` sections define numeric footnotes (``.. [1]``)
# per function. When many functions share one automodule page, those labels
# collide ("Duplicate explicit target name") and the list-only footnotes read
# as "not referenced". Both are cosmetic reST artifacts of the docstring
# idiom, not doc-build errors, so suppress them here to keep ``-W`` meaningful
# for real problems (broken cross-references, missing toctree entries).
# Removing this suppression — by giving each footnote a unique, referenced
# label — is tracked under Roadmap 002 (docstring polish).
suppress_warnings = ["ref.footnote", "docutils"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# windspharm requires a Fortran toolchain and is an optional extra, so mock it
# to keep the docs build (which imports every module) dependency-light.
autodoc_mock_imports = ["windspharm"]

# -- Napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"puffins {version}"
html_theme_options = {
    "github_url": "https://github.com/spencerahill/puffins",
}
