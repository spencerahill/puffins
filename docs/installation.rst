Installation
============

From PyPI
---------

.. code-block:: bash

   pip install puffins

From source (editable)
----------------------

To work on ``puffins`` itself, install it from a clone of the repository in
editable mode:

.. code-block:: bash

   git clone https://github.com/spencerahill/puffins.git
   cd puffins
   pip install -e .

The development and CI environments are managed with
`pixi <https://pixi.prefix.dev/>`_ and locked in ``pixi.lock``. Their packages
come from conda-forge, windspharm included. In a clone, this builds the
default environment and runs the tests:

.. code-block:: bash

   pixi run pytest

Optional dependencies
---------------------

Some functionality in :mod:`puffins.budget_adj` depends on
`windspharm <https://ajdawson.github.io/windspharm/>`_, which requires a
Fortran toolchain and is therefore packaged as an optional extra. The pixi
environments above include it; elsewhere, install it via conda:

.. code-block:: bash

   conda install -c conda-forge windspharm

Coordinate conventions
----------------------

Most functions operate on :class:`xarray.DataArray` objects with standardized
dimension names. Key conventions:

* **Latitude**: degrees, -90 to 90.
* **Pressure levels**: typically Pascal. :func:`puffins.vert_coords.to_pascal`
  converts hPa input to Pa, detecting hPa from magnitude.
* **Streamfunctions**: signed such that counter-clockwise circulation in the
  meridional plane is positive.
* Many calculations assume axisymmetric (zonally-averaged) conditions.
