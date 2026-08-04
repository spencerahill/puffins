"""Type-level contract tests for the ``@overload`` stacks.

Several public functions are declared ``-> ArrayLike`` (the
``DataArray | ndarray | Scalar`` union) but actually return a concrete type
determined by their inputs. ``@overload`` stacks encode that, letting callers
write ``result.shape`` without narrowing. Nothing at runtime exercises those
declarations, so without this file an overload could silently start lying
after a body change or a numpy/xarray upgrade, and mypy would confidently
propagate the wrong type to every caller.

``assert_type`` fails the type check when a call's inferred return type drifts
from the declared one. CI runs ``mypy puffins/``, which covers this directory,
so these assertions are enforced with no extra CI step.

Everything lives under ``TYPE_CHECKING``: mypy analyzes the block, but at
runtime the module imports nothing and does no work. pytest collects the file,
finds no test functions, and moves on.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from typing_extensions import assert_type

    from puffins._typing import Scalar
    from puffins.calculus import lat_circumf, to_radians
    from puffins.dynamics import (
        brunt_vaisala_freq,
        coriolis_param,
        therm_ross_num,
        u_bci_2layer_qg,
    )
    from puffins.grad_bal import grad_wind_bouss
    from puffins.held_hou_1980 import (
        dpot_temp_rce_hh80_dlat,
        hc_edge_hh80_small_angle,
        pot_temp_rce_hh80,
        pot_temp_rce_hh80_small_ang,
        u_crit_switch_lat_hh80,
        u_crit_switch_lat_hh80_small_angle,
        u_rce_hh80,
    )
    from puffins.stats import half_year_symmetry
    from puffins.thermodynamics import (
        dsat_entrop_dtemp_approx,
        exner_func,
        pot_temp,
        pseudoadiabatic_lapse_rate,
        rel_hum_from_temp_dewpoint,
        sat_vap_press_tetens_kelvin,
        specific_humidity,
    )

    _DA: xr.DataArray = xr.DataArray([1.0])
    _ND: np.ndarray = np.array([1.0])
    _SC: Scalar = 1.0

    # ------------------------------------------------------------------
    # Unary: the 3-way split exhausts ArrayLike.
    # ------------------------------------------------------------------
    assert_type(exner_func(_DA), xr.DataArray)
    assert_type(exner_func(_ND), np.ndarray)
    assert_type(exner_func(_SC), Scalar)

    assert_type(specific_humidity(_DA), xr.DataArray)
    assert_type(specific_humidity(_ND), np.ndarray)
    assert_type(specific_humidity(_SC), Scalar)

    assert_type(sat_vap_press_tetens_kelvin(_DA), xr.DataArray)
    assert_type(sat_vap_press_tetens_kelvin(_ND), np.ndarray)
    assert_type(sat_vap_press_tetens_kelvin(_SC), Scalar)

    assert_type(dsat_entrop_dtemp_approx(_DA), xr.DataArray)
    assert_type(dsat_entrop_dtemp_approx(_ND), np.ndarray)
    assert_type(dsat_entrop_dtemp_approx(_SC), Scalar)

    assert_type(coriolis_param(_DA), xr.DataArray)
    assert_type(coriolis_param(_ND), np.ndarray)
    assert_type(coriolis_param(_SC), Scalar)

    assert_type(therm_ross_num(_DA), xr.DataArray)
    assert_type(therm_ross_num(_ND), np.ndarray)
    assert_type(therm_ross_num(_SC), Scalar)

    assert_type(brunt_vaisala_freq(_DA), xr.DataArray)
    assert_type(brunt_vaisala_freq(_ND), np.ndarray)
    assert_type(brunt_vaisala_freq(_SC), Scalar)

    assert_type(u_bci_2layer_qg(_DA), xr.DataArray)
    assert_type(u_bci_2layer_qg(_ND), np.ndarray)
    assert_type(u_bci_2layer_qg(_SC), Scalar)

    assert_type(to_radians(_DA), xr.DataArray)
    assert_type(to_radians(_ND), np.ndarray)
    assert_type(to_radians(_SC), Scalar)

    assert_type(lat_circumf(_DA), xr.DataArray)
    assert_type(lat_circumf(_ND), np.ndarray)
    assert_type(lat_circumf(_SC), Scalar)

    assert_type(u_crit_switch_lat_hh80(_DA), xr.DataArray)
    assert_type(u_crit_switch_lat_hh80(_ND), np.ndarray)
    assert_type(u_crit_switch_lat_hh80(_SC), Scalar)

    assert_type(u_crit_switch_lat_hh80_small_angle(_DA), xr.DataArray)
    assert_type(u_crit_switch_lat_hh80_small_angle(_ND), np.ndarray)
    assert_type(u_crit_switch_lat_hh80_small_angle(_SC), Scalar)

    assert_type(hc_edge_hh80_small_angle(_DA), xr.DataArray)
    assert_type(hc_edge_hh80_small_angle(_ND), np.ndarray)
    assert_type(hc_edge_hh80_small_angle(_SC), Scalar)

    # `delta_h` is a plain float, so `lats` alone drives the return type.
    _DH: float = 1 / 6
    assert_type(dpot_temp_rce_hh80_dlat(_DA, _DH), xr.DataArray)
    assert_type(dpot_temp_rce_hh80_dlat(_ND, _DH), np.ndarray)
    assert_type(dpot_temp_rce_hh80_dlat(_SC, _DH), Scalar)

    # ------------------------------------------------------------------
    # Binary: all 9 input combinations. A DataArray anywhere wins, then an
    # ndarray, and only an all-scalar call stays scalar. These are the cases
    # where an omitted overload silently degrades the result to Any, which
    # would disable checking rather than provide it.
    # ------------------------------------------------------------------
    assert_type(pot_temp(_DA, _DA), xr.DataArray)
    assert_type(pot_temp(_DA, _ND), xr.DataArray)
    assert_type(pot_temp(_DA, _SC), xr.DataArray)
    assert_type(pot_temp(_ND, _DA), xr.DataArray)
    assert_type(pot_temp(_ND, _ND), np.ndarray)
    assert_type(pot_temp(_ND, _SC), np.ndarray)
    assert_type(pot_temp(_SC, _DA), xr.DataArray)
    assert_type(pot_temp(_SC, _ND), np.ndarray)
    assert_type(pot_temp(_SC, _SC), Scalar)

    assert_type(rel_hum_from_temp_dewpoint(_DA, _DA), xr.DataArray)
    assert_type(rel_hum_from_temp_dewpoint(_DA, _ND), xr.DataArray)
    assert_type(rel_hum_from_temp_dewpoint(_DA, _SC), xr.DataArray)
    assert_type(rel_hum_from_temp_dewpoint(_ND, _DA), xr.DataArray)
    assert_type(rel_hum_from_temp_dewpoint(_ND, _ND), np.ndarray)
    assert_type(rel_hum_from_temp_dewpoint(_ND, _SC), np.ndarray)
    assert_type(rel_hum_from_temp_dewpoint(_SC, _DA), xr.DataArray)
    assert_type(rel_hum_from_temp_dewpoint(_SC, _ND), np.ndarray)
    assert_type(rel_hum_from_temp_dewpoint(_SC, _SC), Scalar)

    assert_type(pseudoadiabatic_lapse_rate(_DA, _DA), xr.DataArray)
    assert_type(pseudoadiabatic_lapse_rate(_DA, _ND), xr.DataArray)
    assert_type(pseudoadiabatic_lapse_rate(_DA, _SC), xr.DataArray)
    assert_type(pseudoadiabatic_lapse_rate(_ND, _DA), xr.DataArray)
    assert_type(pseudoadiabatic_lapse_rate(_ND, _ND), np.ndarray)
    assert_type(pseudoadiabatic_lapse_rate(_ND, _SC), np.ndarray)
    assert_type(pseudoadiabatic_lapse_rate(_SC, _DA), xr.DataArray)
    assert_type(pseudoadiabatic_lapse_rate(_SC, _ND), np.ndarray)
    assert_type(pseudoadiabatic_lapse_rate(_SC, _SC), Scalar)

    assert_type(u_rce_hh80(_DA, _DA), xr.DataArray)
    assert_type(u_rce_hh80(_DA, _ND), xr.DataArray)
    assert_type(u_rce_hh80(_DA, _SC), xr.DataArray)
    assert_type(u_rce_hh80(_ND, _DA), xr.DataArray)
    assert_type(u_rce_hh80(_ND, _ND), np.ndarray)
    assert_type(u_rce_hh80(_ND, _SC), np.ndarray)
    assert_type(u_rce_hh80(_SC, _DA), xr.DataArray)
    assert_type(u_rce_hh80(_SC, _ND), np.ndarray)
    assert_type(u_rce_hh80(_SC, _SC), Scalar)

    # `lats` and `z` are the two ArrayLike arguments; the rest are floats.
    _TR: float = 290.0
    _HT: float = 10e3
    _DV: float = 1 / 8
    assert_type(pot_temp_rce_hh80(_DA, _DA, _TR, _HT, _DH, _DV), xr.DataArray)
    assert_type(pot_temp_rce_hh80(_DA, _ND, _TR, _HT, _DH, _DV), xr.DataArray)
    assert_type(pot_temp_rce_hh80(_DA, _SC, _TR, _HT, _DH, _DV), xr.DataArray)
    assert_type(pot_temp_rce_hh80(_ND, _DA, _TR, _HT, _DH, _DV), xr.DataArray)
    assert_type(pot_temp_rce_hh80(_ND, _ND, _TR, _HT, _DH, _DV), np.ndarray)
    assert_type(pot_temp_rce_hh80(_ND, _SC, _TR, _HT, _DH, _DV), np.ndarray)
    assert_type(pot_temp_rce_hh80(_SC, _DA, _TR, _HT, _DH, _DV), xr.DataArray)
    assert_type(pot_temp_rce_hh80(_SC, _ND, _TR, _HT, _DH, _DV), np.ndarray)
    assert_type(pot_temp_rce_hh80(_SC, _SC, _TR, _HT, _DH, _DV), Scalar)

    # Same, but `z` is optional here, so the all-defaults calls must still
    # resolve on `lats` alone rather than matching a DataArray-`z` overload.
    assert_type(pot_temp_rce_hh80_small_ang(_DA, _DA), xr.DataArray)
    assert_type(pot_temp_rce_hh80_small_ang(_DA, _ND), xr.DataArray)
    assert_type(pot_temp_rce_hh80_small_ang(_DA, _SC), xr.DataArray)
    assert_type(pot_temp_rce_hh80_small_ang(_ND, _DA), xr.DataArray)
    assert_type(pot_temp_rce_hh80_small_ang(_ND, _ND), np.ndarray)
    assert_type(pot_temp_rce_hh80_small_ang(_ND, _SC), np.ndarray)
    assert_type(pot_temp_rce_hh80_small_ang(_SC, _DA), xr.DataArray)
    assert_type(pot_temp_rce_hh80_small_ang(_SC, _ND), np.ndarray)
    assert_type(pot_temp_rce_hh80_small_ang(_SC, _SC), Scalar)
    assert_type(pot_temp_rce_hh80_small_ang(_DA), xr.DataArray)
    assert_type(pot_temp_rce_hh80_small_ang(_ND), np.ndarray)
    assert_type(pot_temp_rce_hh80_small_ang(_SC), Scalar)

    _H: float = 10e3
    _T: float = 300.0
    assert_type(grad_wind_bouss(_DA, _H, _T, _DA), xr.DataArray)
    assert_type(grad_wind_bouss(_DA, _H, _T, _ND), xr.DataArray)
    assert_type(grad_wind_bouss(_DA, _H, _T, _SC), xr.DataArray)
    assert_type(grad_wind_bouss(_ND, _H, _T, _DA), xr.DataArray)
    assert_type(grad_wind_bouss(_ND, _H, _T, _ND), np.ndarray)
    assert_type(grad_wind_bouss(_ND, _H, _T, _SC), np.ndarray)
    assert_type(grad_wind_bouss(_SC, _H, _T, _DA), xr.DataArray)
    assert_type(grad_wind_bouss(_SC, _H, _T, _ND), np.ndarray)
    assert_type(grad_wind_bouss(_SC, _H, _T, _SC), Scalar)

    # ------------------------------------------------------------------
    # Two-way split: a 12-month cycle is never a scalar, so `Scalar` is not
    # an accepted input here and the union is DataArray-or-ndarray only.
    # Without these overloads the declared union leaks to every caller, and
    # `.dims` / `.values` on a DataArray result fail to type-check.
    # ------------------------------------------------------------------
    assert_type(half_year_symmetry(_DA), xr.DataArray)
    assert_type(half_year_symmetry(_ND), np.ndarray)
