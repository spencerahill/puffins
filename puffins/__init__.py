"""puffins: tools for large-scale atmospheric and climate dynamics research"""

import contextlib

from . import (
    bootstrap,
    calculus,
    constants,
    dates,
    dynamics,
    eofs,
    eq_area,
    fixed_temp_tropo,
    grad_bal,
    had_cell,
    held_hou_1980,
    hides,
    interp,
    kuo_el,
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
