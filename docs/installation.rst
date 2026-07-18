Installation
============

From source (editable)
----------------------

``puffins`` is not yet published to PyPI. Install it from a clone of the
repository in editable mode:

.. code-block:: bash

   git clone https://github.com/spencerahill/puffins.git
   cd puffins
   pip install -e .

Or, with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv sync

Optional dependencies
---------------------

Some functionality in :mod:`puffins.budget_adj` depends on
`windspharm <https://ajdawson.github.io/windspharm/>`_, which requires a
Fortran toolchain and is therefore packaged as an optional extra. Install it
via conda:

.. code-block:: bash

   conda install -c conda-forge windspharm

Coordinate conventions
----------------------

Most functions operate on :class:`xarray.DataArray` objects with standardized
dimension names. Key conventions:

* **Latitude**: degrees, -90 to 90.
* **Pressure levels**: typically Pascal (many functions accept an
  ``hpa_to_pa`` flag).
* **Streamfunctions**: signed such that counter-clockwise circulation in the
  meridional plane is positive.
* Many calculations assume axisymmetric (zonally-averaged) conditions.
