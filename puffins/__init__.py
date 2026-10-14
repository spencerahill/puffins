"""puffins: tools for large-scale atmospheric and climate dynamics research"""

import contextlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from . import (
    bootstrap,
    calculus,
    constants,
    dates,
    dynamics,
    enso,
    eofs,
    eq_area,
    fixed_temp_tropo,
    grad_bal,
    had_cell,
    held_hou_1980,
    hides,
    interp,
    kuo_el,
    lcl,
    lindzen_hou_1988,
    longitude,
    names,
    nb_utils,
    num_solver,
    plumb_hou_1992,
    polar_amp,
    radiation,
    stats,
    therm_inert,
    thermodynamics,
    tropopause,
    vert_coords,
)
from .longitude import Longitude

with contextlib.suppress(ImportError):  # windspharm not installed
    from . import budget_adj

# The version recorded when puffins was installed. An editable install records
# it once, so a later `git pull` changes the code but not this string.
try:
    __version__ = _version("puffins")
except _PackageNotFoundError:  # imported from a source tree never installed
    __version__ = "0+unknown"
