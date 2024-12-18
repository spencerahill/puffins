"""puffins: tools for large-scale atmospheric and climate dynamics research"""
from . import names
from . import constants
from . import dates
from . import num_solver
from . import longitude
from .longitude import Longitude
from . import calculus
from . import nb_utils
from . import interp
from . import stats
from . import bootstrap

from . import budget_adj
from . import dynamics
from . import grad_bal
from . import eofs
from . import eq_area
from . import fixed_temp_tropo
from . import had_cell
from . import hides
from . import kuo_el
from . import polar_amp
from . import thermodynamics
from . import therm_inert
from . import tropopause
from . import vert_coords

from . import held_hou_1980
from . import lindzen_hou_1988
from . import plumb_hou_1992
