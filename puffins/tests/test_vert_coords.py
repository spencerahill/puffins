"""Tests for vert_coords module."""

import warnings

import numpy as np
import pytest
import xarray as xr

from puffins.constants import GRAV_EARTH, MEAN_SLP_EARTH
from puffins.names import LEV_STR, PFULL_STR, PHALF_STR
from puffins.vert_coords import (
    _flip_dim,
    avg_logp_weighted,
    avg_p_weighted,
    col_avg,
    col_extrema,
    dlogp_from_pfull,
    dlogp_from_phalf,
    dp_from_pfull,
    dp_from_phalf,
    int_dlogp,
    int_dp_g,
    pfull_from_phalf_avg,
    pfull_simm_burr,
    pfull_vals_simm_burr,
    phalf_from_pfull,
    phalf_from_psfc,
    subtract_col_avg,
    to_pascal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pfull_increasing(n: int = 5) -> xr.DataArray:
    """Pressure levels increasing from 100 to 900 hPa (in Pa)."""
    vals = np.linspace(1e4, 9e4, n)
    return xr.DataArray(vals, dims=[LEV_STR], name=LEV_STR)


def _pfull_decreasing(n: int = 5) -> xr.DataArray:
    """Pressure levels decreasing from 900 to 100 hPa (in Pa)."""
    vals = np.linspace(9e4, 1e4, n)
    return xr.DataArray(vals, dims=[LEV_STR], name=LEV_STR)


def _sample_arr_on_pfull(pfull: xr.DataArray) -> xr.DataArray:
    """Sample data array on pressure levels (e.g. temperature-like)."""
    return xr.DataArray(
        np.linspace(200.0, 300.0, pfull.sizes[LEV_STR]),
        dims=[LEV_STR],
        coords={LEV_STR: pfull},
        name="temp",
    )


def _phalf_and_pfull_ref() -> tuple[xr.DataArray, xr.DataArray]:
    """Simple 1-D half-level and full-level reference pressures (increasing)."""
    phalf_vals = np.array([0.0, 2e4, 4e4, 6e4, 8e4, 1e5])
    pfull_vals = 0.5 * (phalf_vals[:-1] + phalf_vals[1:])
    phalf = xr.DataArray(phalf_vals, dims=[PHALF_STR], name=PHALF_STR)
    pfull = xr.DataArray(pfull_vals, dims=[PFULL_STR], name=PFULL_STR)
    return phalf, pfull


# ---------------------------------------------------------------------------
# TestToPascal
# ---------------------------------------------------------------------------


class TestToPascal:
    """Tests for to_pascal."""

    def test_already_pa(self) -> None:
        """Values already in Pa (> 1200) are returned unchanged."""
        arr = np.array([50000.0, 101325.0])
        result = to_pascal(arr)
        np.testing.assert_array_equal(result, arr)

    def test_hpa_converted(self) -> None:
        """Values in hPa (< 1200) are multiplied by 100."""
        arr = np.array([500.0, 1000.0])
        result = to_pascal(arr)
        np.testing.assert_allclose(result, arr * 100.0)

    def test_is_dp_threshold(self) -> None:
        """The is_dp flag lowers the threshold to 400."""
        arr = np.array([350.0])
        # Without is_dp: 350 < 1200 → convert
        result_no_dp = to_pascal(arr, is_dp=False)
        np.testing.assert_allclose(result_no_dp, 35000.0)
        # With is_dp: 350 < 400 → convert
        result_dp = to_pascal(arr, is_dp=True)
        np.testing.assert_allclose(result_dp, 35000.0)

    def test_is_dp_above_threshold(self) -> None:
        """Value above the dp threshold but below the default threshold."""
        arr = np.array([500.0])
        # is_dp=True: 500 >= 400 → no conversion
        result = to_pascal(arr, is_dp=True)
        np.testing.assert_allclose(result, 500.0)

    def test_scalar_input(self) -> None:
        """Works with scalar input."""
        result = to_pascal(500.0)
        assert result == 50000.0

    def test_dataarray_input(self) -> None:
        """Works with xr.DataArray input."""
        arr = xr.DataArray([500.0, 800.0], dims=["x"], name="pressure")
        result = to_pascal(arr)
        xr.testing.assert_allclose(result, xr.DataArray([50000.0, 80000.0], dims=["x"]))

    def test_warn_flag(self) -> None:
        """Warning is issued when warn=True and conversion happens."""
        arr = np.array([500.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_pascal(arr, warn=True)
        assert len(caught) == 1
        assert "hPa -> Pa" in str(caught[0].message)

    def test_no_warn_when_no_conversion(self) -> None:
        """No warning when values are already in Pa."""
        arr = np.array([50000.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_pascal(arr, warn=True)
        assert len(caught) == 0


# ---------------------------------------------------------------------------
# TestIntDpG
# ---------------------------------------------------------------------------


class TestIntDpG:
    """Tests for int_dp_g."""

    def test_uniform_field(self) -> None:
        """Integral of uniform field equals field * dp_sum / g."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = xr.DataArray(
            np.ones(pfull.sizes[LEV_STR]),
            dims=[LEV_STR],
            coords={LEV_STR: pfull},
            name="ones",
        )
        result = int_dp_g(arr, dp)
        expected = dp.sum().item() / GRAV_EARTH
        np.testing.assert_allclose(result.item(), expected)

    def test_scalar_arr(self) -> None:
        """Works with scalar (float) as arr argument."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        result = int_dp_g(1.0, dp)
        expected = dp.sum().item() / GRAV_EARTH
        np.testing.assert_allclose(result.item(), expected)

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = _sample_arr_on_pfull(pfull)
        result = int_dp_g(arr, dp)
        assert isinstance(result, xr.DataArray)

    def test_custom_grav(self) -> None:
        """Custom gravity constant scales the result."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = _sample_arr_on_pfull(pfull)
        result_default = int_dp_g(arr, dp)
        result_half_g = int_dp_g(arr, dp, grav=GRAV_EARTH / 2)
        np.testing.assert_allclose(result_half_g.item(), 2.0 * result_default.item())


# ---------------------------------------------------------------------------
# TestIntDlogp
# ---------------------------------------------------------------------------


class TestIntDlogp:
    """Tests for int_dlogp."""

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        arr = _sample_arr_on_pfull(pfull)
        result = int_dlogp(arr)
        assert isinstance(result, xr.DataArray)

    def test_positive_for_positive_field(self) -> None:
        """Integral of a positive field with positive dlogp is positive."""
        pfull = _pfull_increasing()
        arr = _sample_arr_on_pfull(pfull)
        result = int_dlogp(arr)
        assert result.item() > 0


# ---------------------------------------------------------------------------
# TestColAvg
# ---------------------------------------------------------------------------


class TestColAvg:
    """Tests for col_avg."""

    def test_uniform_field(self) -> None:
        """Column average of a uniform field is that constant."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = xr.DataArray(
            5.0 * np.ones(pfull.sizes[LEV_STR]),
            dims=[LEV_STR],
            coords={LEV_STR: pfull},
            name="const",
        )
        result = col_avg(arr, dp)
        np.testing.assert_allclose(result.item(), 5.0)

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = _sample_arr_on_pfull(pfull)
        result = col_avg(arr, dp)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestSubtractColAvg
# ---------------------------------------------------------------------------


class TestSubtractColAvg:
    """Tests for subtract_col_avg."""

    def test_result_has_zero_weighted_mean(self) -> None:
        """After subtracting column average, the weighted integral is ~zero."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = _sample_arr_on_pfull(pfull)
        result = subtract_col_avg(arr, dp)
        weighted_sum = int_dp_g(result, dp)
        np.testing.assert_allclose(weighted_sum.item(), 0.0, atol=1e-10)

    def test_preserves_shape(self) -> None:
        """Result has same shape as input."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = _sample_arr_on_pfull(pfull)
        result = subtract_col_avg(arr, dp)
        assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# TestPhalfFromPfull
# ---------------------------------------------------------------------------


class TestPhalfFromPfull:
    """Tests for phalf_from_pfull."""

    def test_increasing_pressure(self) -> None:
        """Half levels bracket full levels when pressure increases."""
        pfull = _pfull_increasing()
        phalf = phalf_from_pfull(pfull)
        # First half level is p_top (0.0 by default)
        np.testing.assert_allclose(phalf.values[0], 0.0)
        # Last half level is p_bot (MEAN_SLP_EARTH by default)
        np.testing.assert_allclose(phalf.values[-1], MEAN_SLP_EARTH)

    def test_decreasing_pressure(self) -> None:
        """Half levels bracket full levels when pressure decreases."""
        pfull = _pfull_decreasing()
        phalf = phalf_from_pfull(pfull)
        # First half level is p_bot, last is p_top (reversed)
        np.testing.assert_allclose(phalf.values[0], MEAN_SLP_EARTH)
        np.testing.assert_allclose(phalf.values[-1], 0.0)

    def test_length(self) -> None:
        """Half levels have one more element than full levels."""
        pfull = _pfull_increasing(n=7)
        phalf = phalf_from_pfull(pfull)
        assert phalf.sizes[PHALF_STR] == 8

    def test_inner_values_are_averages(self) -> None:
        """Inner half levels are averages of adjacent full levels."""
        pfull = _pfull_increasing()
        phalf = phalf_from_pfull(pfull)
        for i in range(1, len(phalf) - 1):
            expected = 0.5 * (pfull.values[i - 1] + pfull.values[i])
            np.testing.assert_allclose(phalf.values[i], expected)

    def test_custom_bounds(self) -> None:
        """Custom p_top and p_bot are respected."""
        pfull = _pfull_increasing()
        phalf = phalf_from_pfull(pfull, p_top=100.0, p_bot=1e5)
        np.testing.assert_allclose(phalf.values[0], 100.0)
        np.testing.assert_allclose(phalf.values[-1], 1e5)

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        result = phalf_from_pfull(pfull)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestDpFromPfull
# ---------------------------------------------------------------------------


class TestDpFromPfull:
    """Tests for dp_from_pfull."""

    def test_all_positive(self) -> None:
        """All dp values are positive."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        assert (dp.values > 0).all()

    def test_same_length_as_pfull(self) -> None:
        """dp has the same length as pfull."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        assert dp.sizes[LEV_STR] == pfull.sizes[LEV_STR]

    def test_sum_equals_p_range(self) -> None:
        """Sum of dp equals p_bot - p_top."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        np.testing.assert_allclose(dp.sum().item(), MEAN_SLP_EARTH - 0.0)

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        result = dp_from_pfull(pfull)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestDpFromPhalf
# ---------------------------------------------------------------------------


class TestDpFromPhalf:
    """Tests for dp_from_phalf."""

    def test_all_positive(self) -> None:
        """All dp values are positive."""
        phalf, pfull = _phalf_and_pfull_ref()
        dp = dp_from_phalf(phalf, pfull)
        assert (dp.values > 0).all()

    def test_length_matches_pfull(self) -> None:
        """dp has one fewer element than phalf (= len of pfull)."""
        phalf, pfull = _phalf_and_pfull_ref()
        dp = dp_from_phalf(phalf, pfull)
        assert dp.sizes[PFULL_STR] == pfull.sizes[PFULL_STR]

    def test_sum_equals_total_pressure_range(self) -> None:
        """Sum of dp equals total pressure range."""
        phalf, pfull = _phalf_and_pfull_ref()
        dp = dp_from_phalf(phalf, pfull)
        np.testing.assert_allclose(dp.sum().item(), phalf.values[-1] - phalf.values[0])

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, pfull = _phalf_and_pfull_ref()
        result = dp_from_phalf(phalf, pfull)
        assert isinstance(result, xr.DataArray)

    def test_named_dp(self) -> None:
        """Result is named 'dp'."""
        phalf, pfull = _phalf_and_pfull_ref()
        dp = dp_from_phalf(phalf, pfull)
        assert dp.name == "dp"


# ---------------------------------------------------------------------------
# TestDlogpFromPhalf
# ---------------------------------------------------------------------------


class TestDlogpFromPhalf:
    """Tests for dlogp_from_phalf."""

    def test_positive_values(self) -> None:
        """dlogp values are positive for increasing pressure."""
        phalf, pfull = _phalf_and_pfull_ref()
        # Skip top level with p=0 — tested separately
        phalf_nonzero = phalf.isel({PHALF_STR: slice(1, None)})
        pfull_nonzero = pfull.isel({PFULL_STR: slice(1, None)})
        dlogp = dlogp_from_phalf(phalf_nonzero, pfull_nonzero)
        assert (dlogp.values > 0).all()

    def test_zero_top_handled(self) -> None:
        """Top pressure of zero is replaced to avoid log(0)."""
        phalf, pfull = _phalf_and_pfull_ref()
        # phalf[0] == 0, so it should be replaced with 0.5 * phalf[1]
        dlogp = dlogp_from_phalf(phalf, pfull)
        assert np.all(np.isfinite(dlogp.values))

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, pfull = _phalf_and_pfull_ref()
        result = dlogp_from_phalf(phalf, pfull)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestDlogpFromPfull
# ---------------------------------------------------------------------------


class TestDlogpFromPfull:
    """Tests for dlogp_from_pfull."""

    def test_finite_values(self) -> None:
        """All dlogp values are finite."""
        pfull = _pfull_increasing()
        dlogp = dlogp_from_pfull(pfull)
        assert np.all(np.isfinite(dlogp.values))

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        result = dlogp_from_pfull(pfull)
        assert isinstance(result, xr.DataArray)

    def test_same_length_as_pfull(self) -> None:
        """dlogp has the same length as pfull."""
        pfull = _pfull_increasing()
        dlogp = dlogp_from_pfull(pfull)
        assert dlogp.sizes[LEV_STR] == pfull.sizes[LEV_STR]


# ---------------------------------------------------------------------------
# TestPhalfFromPsfc
# ---------------------------------------------------------------------------


class TestPhalfFromPsfc:
    """Tests for phalf_from_psfc."""

    def test_pure_sigma(self) -> None:
        """With pk=0, phalf = p_sfc * bk."""
        bk = np.array([0.0, 0.5, 1.0])
        pk = np.array([0.0, 0.0, 0.0])
        p_sfc = 1e5
        result = phalf_from_psfc(bk, pk, p_sfc)
        expected = np.array([0.0, 5e4, 1e5])
        np.testing.assert_allclose(result, expected)

    def test_pure_pressure(self) -> None:
        """With bk=0, phalf = pk."""
        bk = np.array([0.0, 0.0, 0.0])
        pk = np.array([0.0, 5e4, 1e5])
        p_sfc = 1e5
        result = phalf_from_psfc(bk, pk, p_sfc)
        np.testing.assert_allclose(result, pk)

    def test_hybrid(self) -> None:
        """Hybrid sigma-pressure coordinates."""
        bk = np.array([0.0, 0.3, 0.7])
        pk = np.array([0.0, 1e4, 2e4])
        p_sfc = 1e5
        result = phalf_from_psfc(bk, pk, p_sfc)
        expected = p_sfc * bk + pk
        np.testing.assert_allclose(result, expected)

    def test_dataarray_inputs(self) -> None:
        """Works with DataArray inputs."""
        bk = xr.DataArray([0.0, 0.5, 1.0], dims=[PHALF_STR])
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=[PHALF_STR])
        p_sfc = xr.DataArray(1e5)
        result = phalf_from_psfc(bk, pk, p_sfc)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestPfullFromPhalfAvg
# ---------------------------------------------------------------------------


class TestPfullFromPhalfAvg:
    """Tests for pfull_from_phalf_avg."""

    def test_values_between_half_levels(self) -> None:
        """Full-level pressures lie between adjacent half-level pressures."""
        phalf, pfull_ref = _phalf_and_pfull_ref()
        pfull = pfull_from_phalf_avg(phalf, pfull_ref)
        for i in range(len(pfull_ref)):
            assert phalf.values[i] <= pfull.values[i] <= phalf.values[i + 1]

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, pfull_ref = _phalf_and_pfull_ref()
        result = pfull_from_phalf_avg(phalf, pfull_ref)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestPfullValsSimBurr
# ---------------------------------------------------------------------------


class TestPfullValsSimBurr:
    """Tests for pfull_vals_simm_burr."""

    def _make_inputs(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Create valid inputs for Simmons-Burridge calculation."""
        phalf_vals = np.array([100.0, 5000.0, 20000.0, 50000.0, 80000.0, 101325.0])
        pfull_vals = 0.5 * (phalf_vals[:-1] + phalf_vals[1:])
        phalf = xr.DataArray(phalf_vals, dims=[PHALF_STR])
        phalf_ref = phalf.copy()
        pfull_ref = xr.DataArray(pfull_vals, dims=[PFULL_STR])
        return phalf, phalf_ref, pfull_ref

    def test_returns_ndarray(self) -> None:
        """Return type is numpy ndarray."""
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        result = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref)
        assert isinstance(result, np.ndarray)

    def test_length(self) -> None:
        """Output has one fewer element than phalf."""
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        result = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref)
        assert len(result) == len(phalf) - 1

    def test_values_positive(self) -> None:
        """All computed pressures are positive."""
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        result = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref)
        assert (result > 0).all()

    def test_values_between_half_levels(self) -> None:
        """Full-level pressures lie between adjacent half-level pressures."""
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        result = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref)
        for i in range(len(result)):
            assert phalf.values[i] <= result[i] <= phalf.values[i + 1]


# ---------------------------------------------------------------------------
# TestPfullSimBurr
# ---------------------------------------------------------------------------


class TestPfullSimBurr:
    """Tests for pfull_simm_burr.

    pfull_simm_burr uses dp_from_phalf internally, which requires phalf
    to have at least one dimension beyond phalf (e.g. lat or time) so that
    the template broadcasting works correctly with xr.concat.
    """

    def _make_inputs(
        self, increasing: bool = True
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Create valid 2-D inputs for Simmons-Burridge calculation.

        Adds a 'time' dimension so dp_from_phalf broadcasting and xr.concat
        work correctly.
        """
        phalf_vals_1d = np.array([100.0, 5000.0, 20000.0, 50000.0, 80000.0, 101325.0])
        if not increasing:
            phalf_vals_1d = phalf_vals_1d[::-1]
        sorted_ph = np.sort(phalf_vals_1d)
        pfull_vals_1d = 0.5 * (sorted_ph[:-1] + sorted_ph[1:])
        if not increasing:
            pfull_vals_1d = pfull_vals_1d[::-1]

        # Reference arrays are 1-D
        phalf_ref = xr.DataArray(phalf_vals_1d, dims=[PHALF_STR])
        pfull_ref = xr.DataArray(pfull_vals_1d, dims=[PFULL_STR])

        # Actual phalf is 2-D (phalf x time) — identical across time
        ntimes = 2
        phalf_2d = xr.DataArray(
            np.tile(phalf_vals_1d[:, np.newaxis], (1, ntimes)),
            dims=[PHALF_STR, "time"],
            coords={"time": np.arange(ntimes)},
        )
        return phalf_2d, phalf_ref, pfull_ref

    @pytest.mark.xfail(
        reason="Pre-existing xr.concat bug (#26): pfull_top has 'pfull' as "
        "scalar coordinate not dimension; modern xarray raises ValueError.",
        strict=True,
    )
    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.xfail(
        reason="Pre-existing xr.concat bug (#26): pfull_top has 'pfull' as "
        "scalar coordinate not dimension; modern xarray raises ValueError.",
        strict=True,
    )
    def test_increasing_pressure(self) -> None:
        """Works with increasing pressure ordering."""
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=True)
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert result.sizes[PFULL_STR] == pfull_ref.sizes[PFULL_STR]
        assert (result.values > 0).all()

    @pytest.mark.xfail(
        reason="Pre-existing xr.concat bug (#26): pfull_top has 'pfull' as "
        "scalar coordinate not dimension; modern xarray raises ValueError.",
        strict=True,
    )
    def test_decreasing_pressure(self) -> None:
        """Works with decreasing pressure ordering."""
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=False)
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert result.sizes[PFULL_STR] == pfull_ref.sizes[PFULL_STR]
        assert (result.values > 0).all()


# ---------------------------------------------------------------------------
# TestFlipDim
# ---------------------------------------------------------------------------


class TestFlipDim:
    """Tests for _flip_dim."""

    def test_reverses_values(self) -> None:
        """Flipping reverses the values along the dimension."""
        arr = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["x"], name="test")
        result = _flip_dim(arr, "x")
        np.testing.assert_array_equal(result.values, [4.0, 3.0, 2.0, 1.0])

    def test_double_flip_identity(self) -> None:
        """Flipping twice returns the original."""
        arr = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], name="test")
        result = _flip_dim(_flip_dim(arr, "x"), "x")
        np.testing.assert_array_equal(result.values, arr.values)

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        arr = xr.DataArray([1.0, 2.0], dims=["x"], name="test")
        result = _flip_dim(arr, "x")
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestAvgPWeighted
# ---------------------------------------------------------------------------


class TestAvgPWeighted:
    """Tests for avg_p_weighted."""

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, pfull = _phalf_and_pfull_ref()
        arr = xr.DataArray(
            np.ones(pfull.sizes[PFULL_STR]),
            dims=[PFULL_STR],
            coords={PFULL_STR: pfull},
            name="ones",
        )
        # avg_p_weighted uses dp_from_phalf which expects phalf_str=PHALF_STR
        result = avg_p_weighted(arr, phalf, pfull, p_str=PFULL_STR)
        assert isinstance(result, xr.DataArray)

    def test_uniform_field(self) -> None:
        """Average of a uniform field is that constant at every level."""
        phalf, pfull = _phalf_and_pfull_ref()
        arr = xr.DataArray(
            7.0 * np.ones(pfull.sizes[PFULL_STR]),
            dims=[PFULL_STR],
            coords={PFULL_STR: pfull},
            name="const",
        )
        result = avg_p_weighted(arr, phalf, pfull, p_str=PFULL_STR)
        np.testing.assert_allclose(result.values, 7.0)


# ---------------------------------------------------------------------------
# TestAvgLogpWeighted
# ---------------------------------------------------------------------------


class TestAvgLogpWeighted:
    """Tests for avg_logp_weighted."""

    def _make_inputs(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Non-zero phalf and pfull for log-pressure weighting."""
        phalf_vals = np.array([1000.0, 2e4, 4e4, 6e4, 8e4, 1e5])
        pfull_vals = 0.5 * (phalf_vals[:-1] + phalf_vals[1:])
        phalf = xr.DataArray(phalf_vals, dims=[PHALF_STR])
        pfull = xr.DataArray(pfull_vals, dims=[PFULL_STR])
        return phalf, pfull

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, pfull = self._make_inputs()
        arr = xr.DataArray(
            np.ones(pfull.sizes[PFULL_STR]),
            dims=[PFULL_STR],
            coords={PFULL_STR: pfull},
            name="ones",
        )
        result = avg_logp_weighted(arr, phalf, pfull, p_str=PFULL_STR)
        assert isinstance(result, xr.DataArray)

    def test_uniform_field(self) -> None:
        """Average of a uniform field is that constant."""
        phalf, pfull = self._make_inputs()
        arr = xr.DataArray(
            3.0 * np.ones(pfull.sizes[PFULL_STR]),
            dims=[PFULL_STR],
            coords={PFULL_STR: pfull},
            name="const",
        )
        result = avg_logp_weighted(arr, phalf, pfull, p_str=PFULL_STR)
        np.testing.assert_allclose(result.values, 3.0)


# ---------------------------------------------------------------------------
# TestColExtrema
# ---------------------------------------------------------------------------


class TestColExtrema:
    """Tests for col_extrema."""

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        pfull = _pfull_increasing()
        arr = xr.DataArray(
            [1.0, 3.0, 2.0, 4.0, 1.0],
            dims=[LEV_STR],
            coords={LEV_STR: pfull},
            name="test",
        )
        result = col_extrema(arr)
        assert isinstance(result, xr.DataArray)

    def test_monotonic_has_no_extrema(self) -> None:
        """A monotonically increasing field has no internal extrema."""
        pfull = _pfull_increasing()
        arr = _sample_arr_on_pfull(pfull)  # linearly increasing
        result = col_extrema(arr)
        # All values should be NaN (no sign changes in derivative)
        assert np.all(np.isnan(result.values))

    def test_detects_maximum(self) -> None:
        """Detects a local maximum in the column."""
        pfull = _pfull_increasing()
        # Create a profile with a peak in the middle
        arr = xr.DataArray(
            [1.0, 3.0, 5.0, 3.0, 1.0],
            dims=[LEV_STR],
            coords={LEV_STR: pfull},
            name="peaked",
        )
        result = col_extrema(arr)
        # Non-NaN values should exist near the peak
        assert not np.all(np.isnan(result.values))
