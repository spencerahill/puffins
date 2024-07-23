# Parameters
T_TRIP = pf.constants.T_TRIP
P_TRIP = pf.constants.P_TRIP
E_0V = pf.constants.E_0V
E_0S = pf.constants.E_0S
R_D = pf.constants.R_D
R_V = pf.constants.R_V
C_VV = pf.constants.C_VV
C_VL = pf.constants.C_VL
C_PD = pf.constants.C_PD
C_PV = pf.constants.C_PV
GRAV = pf.constants.GRAV_EARTH


def sat_vap_press_liq_wat(
    temp, 
    p_trip=P_TRIP, 
    t_trip=T_TRIP, 
    r_d=R_D, 
    e_0v=E_0V,
    r_v=R_V,
    c_pv=C_PV,
    c_vv=C_VV,
    c_vl=C_VL,
):
    """Saturation vapor pressure over liquid water.  Romps 2017."""
    return (p_trip * (temp / t_trip) ** ((c_pv - c_vl) / r_v) *
            np.exp((e_0v - (c_vv - c_vl) * t_trip) / r_v * (1 / t_trip - 1 / temp)))


def _q_v(press, vap_press, r_d=R_D, r_v=R_V):
    return r_d * vap_press / (r_v * press + (r_d - r_v) * vap_press)


def gas_const_moist_air(press, vap_press, r_d=R_D, r_v=R_V):
    """Gas constant of moist air.  Romps 2017."""
    q_v = _q_v(press, vap_press, r_d=r_d, r_v=r_v)
    return (1 - q_v) * r_d + q_v * r_v


def spec_heat_const_press_moist_air(press, vap_press, r_d=R_D, r_v=R_V, c_pd=C_PD, c_pv=C_PV):
    """Specifi heat at constant pressure of moist air."""
    q_v = _q_v(press, vap_press, r_d=r_d, r_v=r_v)
    return (1 - q_v) * c_pd + q_v * c_pv


def lift_cond_temp(
    press, 
    temp, 
    rel_hum, 
    z_0=0.,
    p_trip=P_TRIP, 
    t_trip=T_TRIP, 
    r_d=R_D,
    r_v=R_V,
    c_pd=C_PD,
    c_pv=C_PV,
    c_vv=C_VV,
    c_vl=C_VL,
    e_0v=E_0V,
    grav=GRAV,
):
    """Temperature at lifting condensation level.  C.f. Romps 2017.
    
    Romps, David M. 2017. “Exact Expression for the Lifting Condensation Level.” 
    Journal of the Atmospheric Sciences 74 (12): 3891–3900. 
    https://doi.org/10.1175/JAS-D-17-0102.1.

    """
    sat_vap_press = sat_vap_press_liq_wat(temp)
    vap_press = rel_hum * sat_vap_press

    r_m = gas_const_moist_air(press, vap_press, r_d=r_d, r_v=r_v)
    c_pm = spec_heat_const_press_moist_air(
        press, vap_press, r_d=r_d, r_v=r_v, c_pd=c_pd, c_pv=c_pv)

    term_a = -(c_pv - c_vl) / r_v + c_pm / r_m
    term_b = -(e_0v - (c_vv - c_vl) * t_trip) / (r_v * temp)
    term_c = vap_press / sat_vap_press * np.exp(
        -(e_0v - (c_vv - c_vl) * t_trip) / (r_v * temp))

    return z_0 + c_pm * temp / grav * (1. -
       term_b / (term_a * scipy.special.lambertw(
           term_b / term_a * term_c ** (1. / term_a), -1).real))


def temp_lift_cond_level(
    press, 
    temp, 
    rel_hum, 
    p_trip=P_TRIP, 
    t_trip=T_TRIP, 
    r_d=R_D,
    r_v=R_V,
    c_pd=C_PD,
    c_pv=C_PV,
    c_vv=C_VV,
    c_vl=C_VL,
    e_0v=E_0V,
):
    """Temperature at lifting condensation level.  C.f. Romps 2017.
    
    Romps, David M. 2017. “Exact Expression for the Lifting Condensation Level.” 
    Journal of the Atmospheric Sciences 74 (12): 3891–3900. 
    https://doi.org/10.1175/JAS-D-17-0102.1.
    
    Modified from scripts provided by D. Romps at
    https://romps.berkeley.edu/papers/pubdata/2016/lcl/lcl.py

    """
    sat_vap_press = sat_vap_press_liq_wat(temp)
    vap_press = rel_hum * sat_vap_press

    r_m = gas_const_moist_air(press, vap_press, r_d=r_d, r_v=r_v)
    c_pm = spec_heat_const_press_moist_air(
        press, vap_press, r_d=r_d, r_v=r_v, c_pd=c_pd, c_pv=c_pv)

    term_a = (c_vl - c_pv) / r_v + c_pm / r_m
    term_b = -(e_0v - (c_vv - c_vl) * t_trip) / (r_v * temp)
    term_c = term_b / term_a
    return term_c * temp / scipy.special.lambertw(
        rel_hum ** (1. / term_a) * term_c * np.exp(term_c), -1).real


def lift_cond_level(
    press, 
    temp, 
    rel_hum, 
    z_0=0.,
    p_trip=P_TRIP, 
    t_trip=T_TRIP, 
    r_d=R_D,
    r_v=R_V,
    c_pd=C_PD,
    c_pv=C_PV,
    c_vv=C_VV,
    c_vl=C_VL,
    e_0v=E_0V,
    grav=GRAV,
):
    """Lifting condensation level.  C.f. Romps 2017.
    
    Romps, David M. 2017. “Exact Expression for the Lifting Condensation Level.” 
    Journal of the Atmospheric Sciences 74 (12): 3891–3900. 
    https://doi.org/10.1175/JAS-D-17-0102.1.
    
    Modified from scripts provided by D. Romps at
    https://romps.berkeley.edu/papers/pubdata/2016/lcl/lcl.py

    """
    sat_vap_press = sat_vap_press_liq_wat(temp)
    vap_press = rel_hum * sat_vap_press
    c_pm = spec_heat_const_press_moist_air(
        press, vap_press, r_d=r_d, r_v=r_v, c_pd=c_pd, c_pv=c_pv)
    temp_lcl = temp_lift_cond_level(
        press, 
        temp, 
        rel_hum, 
        p_trip=p_trip, 
        t_trip=t_trip, 
        r_d=r_d,
        r_v=r_v,
        c_pd=c_pd,
        c_pv=c_pv,
        c_vv=c_vv,
        c_vl=c_vl,
        e_0v=e_0v,
    )
    return z_0 + c_pm / grav * (temp - temp_lcl)
