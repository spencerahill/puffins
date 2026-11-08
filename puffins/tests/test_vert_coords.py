"""Tests for vert_coords module."""

import warnings
from collections.abc import Callable

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
    pfull_from_phalf_log_mean,
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

    def test_below_both_thresholds(self) -> None:
        """A value below both thresholds is converted regardless of is_dp."""
        arr = np.array([350.0])
        np.testing.assert_allclose(to_pascal(arr, is_dp=False), 35000.0)
        np.testing.assert_allclose(to_pascal(arr, is_dp=True), 35000.0)

    def test_is_dp_above_threshold(self) -> None:
        """Value above the dp threshold but below the default threshold."""
        arr = np.array([500.0])
        # is_dp=True: 500 >= 400 → no conversion
        result = to_pascal(arr, is_dp=True)
        np.testing.assert_allclose(result, 500.0)

    def test_default_threshold_boundary(self) -> None:
        """Strictly-less-than check at 1200: 1199.9 converts, 1200.0 does not."""
        np.testing.assert_allclose(to_pascal(np.array([1199.9])), 119990.0)
        np.testing.assert_allclose(to_pascal(np.array([1200.0])), 1200.0)
        np.testing.assert_allclose(to_pascal(np.array([1200.1])), 1200.1)

    def test_is_dp_threshold_boundary(self) -> None:
        """Strictly-less-than check at 400: 399.9 converts, 400.0 does not."""
        np.testing.assert_allclose(to_pascal(np.array([399.9]), is_dp=True), 39990.0)
        np.testing.assert_allclose(to_pascal(np.array([400.0]), is_dp=True), 400.0)
        np.testing.assert_allclose(to_pascal(np.array([400.1]), is_dp=True), 400.1)

    def test_small_pa_false_positive(self) -> None:
        """Genuinely-small Pa values (e.g. upper-atmosphere ~600 Pa) are
        incorrectly treated as hPa — documents the known heuristic limitation
        called out in the docstring.
        """
        arr = np.array([600.0])  # 600 Pa at e.g. ~6 hPa stratosphere
        result = to_pascal(arr)
        np.testing.assert_allclose(result, 60000.0)

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

    def test_nan_levels(self) -> None:
        """NaN levels are skipped; a column with no finite level is NaN (#72).

        Three columns: all levels finite, the middle level NaN (as for a level
        below ground), and every level NaN.  The last returned 0 before #72.
        """
        dp_vals = np.array([100.0, 200.0, 300.0])
        vals = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
        plev = [1e4, 5e4, 9e4]
        dp = xr.DataArray(dp_vals, dims=[LEV_STR], coords={LEV_STR: plev})
        arr = xr.DataArray(
            vals, dims=["col", LEV_STR], coords={LEV_STR: plev}, name="arr"
        )
        result = int_dp_g(arr, dp)
        expected = np.array(
            [
                (1.0 * 100.0 + 2.0 * 200.0 + 3.0 * 300.0) / GRAV_EARTH,
                (1.0 * 100.0 + 3.0 * 300.0) / GRAV_EARTH,
                np.nan,
            ]
        )
        assert np.isnan(result.values[2])
        np.testing.assert_allclose(result.values, expected)

    def test_single_finite_level_kept(self) -> None:
        """One finite level suffices: the column gives that level, not NaN.

        As for a high-topography column where only the top level is above
        ground.
        """
        plev = [1e4, 5e4, 9e4]
        dp = xr.DataArray([100.0, 200.0, 300.0], dims=[LEV_STR], coords={LEV_STR: plev})
        arr = xr.DataArray(
            [np.nan, np.nan, 3.0], dims=[LEV_STR], coords={LEV_STR: plev}, name="arr"
        )
        np.testing.assert_allclose(int_dp_g(arr, dp).item(), 3.0 * 300.0 / GRAV_EARTH)

    def test_nan_in_dp(self) -> None:
        """NaN in dp is skipped like NaN in arr; an all-NaN dp column is NaN."""
        plev = [1e4, 5e4, 9e4]
        dp = xr.DataArray(
            [[100.0, 200.0, 300.0], [100.0, np.nan, 300.0], [np.nan, np.nan, np.nan]],
            dims=["col", LEV_STR],
            coords={LEV_STR: plev},
        )
        arr = xr.DataArray(
            np.full((3, 3), 2.0),
            dims=["col", LEV_STR],
            coords={LEV_STR: plev},
            name="arr",
        )
        np.testing.assert_allclose(
            int_dp_g(arr, dp).values, np.array([1200.0, 800.0, np.nan]) / GRAV_EARTH
        )
        # Scalar arr, the path col_avg uses for its denominator.
        np.testing.assert_allclose(
            int_dp_g(1.0, dp).values, np.array([600.0, 400.0, np.nan]) / GRAV_EARTH
        )


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

    def test_all_nan_column_is_nan(self) -> None:
        """A column with no finite level averages to NaN, not 0 (#72)."""
        pfull = _pfull_increasing()
        dp = dp_from_pfull(pfull)
        arr = xr.DataArray(
            np.full(pfull.sizes[LEV_STR], np.nan),
            dims=[LEV_STR],
            coords={LEV_STR: pfull},
            name="missing",
        )
        assert np.isnan(col_avg(arr, dp).item())

    def test_masked_level_left_out_of_the_thickness(self) -> None:
        """A NaN level of arr is left out of the column's thickness too (#82).

        Five levels 200 hPa thick, arr 3 with the bottom level NaN, as below
        ground, and dp unmasked: the mean over the four defined levels is 3.
        Summing the thickness over all five levels gave 3 * 4 / 5 = 2.4.
        """
        plev = [1e4, 3e4, 5e4, 7e4, 9e4]
        dp = xr.DataArray(np.full(5, 2e4), dims=[LEV_STR], coords={LEV_STR: plev})
        arr = xr.DataArray(
            [3.0, 3.0, 3.0, 3.0, np.nan],
            dims=[LEV_STR],
            coords={LEV_STR: plev},
            name="arr",
        )
        np.testing.assert_allclose(col_avg(arr, dp).item(), 3.0, rtol=1e-14)

    @pytest.mark.parametrize("mask_dp", [False, True], ids=["dp_unmasked", "dp_masked"])
    def test_each_column_over_its_own_defined_levels(self, mask_dp: bool) -> None:
        """Each column is averaged over the levels where it is defined.

        The three columns are defined at all five levels, at the top four,
        and at the second and third, against an unmasked dp of 100 to 500
        Pa, or dp masked like arr, which gives the same averages.
        """
        plev = [1e4, 3e4, 5e4, 7e4, 9e4]
        dp_vals = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        vals = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, np.nan],
                [np.nan, 2.0, 3.0, np.nan, np.nan],
            ]
        )
        dp = xr.DataArray(dp_vals, dims=[LEV_STR], coords={LEV_STR: plev})
        arr = xr.DataArray(
            vals, dims=["col", LEV_STR], coords={LEV_STR: plev}, name="arr"
        )
        if mask_dp:
            dp = dp.where(arr.notnull())
        defined = np.isfinite(vals)
        expected = np.nansum(vals * dp_vals, axis=1) / (defined * dp_vals).sum(axis=1)
        np.testing.assert_allclose(expected, [5500 / 1500, 3000 / 1000, 1300 / 500])
        np.testing.assert_allclose(col_avg(arr, dp).values, expected, rtol=1e-14)


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

    def test_masked_level_integrates_to_zero(self) -> None:
        """With arr masked below ground and dp not, the defined levels sum to 0 (#82).

        Five levels 200 hPa thick, arr 1 to 4 with the bottom level NaN: the
        column average over the four defined levels is 2.5. Averaging over all
        five thicknesses gave 2, which left a column integral of 2 times the
        thickness of one level.
        """
        plev = [1e4, 3e4, 5e4, 7e4, 9e4]
        dp = xr.DataArray(np.full(5, 2e4), dims=[LEV_STR], coords={LEV_STR: plev})
        arr = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, np.nan],
            dims=[LEV_STR],
            coords={LEV_STR: plev},
            name="arr",
        )
        result = subtract_col_avg(arr, dp)
        np.testing.assert_allclose(
            result.values[:4], [-1.5, -0.5, 0.5, 1.5], rtol=1e-14
        )
        assert np.isnan(result.values[4])
        np.testing.assert_allclose(int_dp_g(result, dp).item(), 0.0, atol=1e-10)


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

    @pytest.mark.parametrize("increasing", [True, False])
    def test_mean_of_bounding_half_levels(self, increasing: bool) -> None:
        """Each full level is the mean of its two half levels, in either order.

        With pressure decreasing along ``phalf``, the first half level of each
        pair is the bottom of the layer. The function used to add half the
        layer's thickness to it, placing every full level below its layer.
        """
        column = np.array([100.0, 5000.0, 20000.0, 50000.0, 80000.0, 101325.0])
        # Not proportional to the first column, so no column is a scaled copy.
        phalf_vals = np.stack([column, 0.9 * column + 300.0])
        if not increasing:
            phalf_vals = phalf_vals[:, ::-1]
        phalf = xr.DataArray(
            phalf_vals, dims=["lat", PHALF_STR], coords={"lat": [0.0, 30.0]}
        )
        pfull_ref = xr.DataArray(np.arange(5.0), dims=[PFULL_STR])
        result = pfull_from_phalf_avg(phalf, pfull_ref)
        assert result.dims == ("lat", PFULL_STR)
        assert result.name == PFULL_STR
        np.testing.assert_allclose(
            result.values, 0.5 * (phalf_vals[:, :-1] + phalf_vals[:, 1:]), rtol=1e-15
        )


# ---------------------------------------------------------------------------
# TestPfullFromPhalfLogMean
# ---------------------------------------------------------------------------


class TestPfullFromPhalfLogMean:
    """Tests for pfull_from_phalf_log_mean."""

    def test_emars_top_layer(self) -> None:
        """EMARS's own full-level pressure for its top layer.

        Issue #118 quotes EMARS's top half levels, 0.0002 and 0.00057381 hPa,
        and its stored top full level, 0.00035467 hPa, to five significant
        figures; the log-mean of the first two is 1.45e-5 below the third. The
        arithmetic mean is 9.1% above it and the geometric mean 4.5% below.
        """
        phalf = xr.DataArray([0.02, 0.057381], dims=[PHALF_STR])
        pfull_ref = xr.DataArray([0.0], dims=[PFULL_STR])
        result = pfull_from_phalf_log_mean(phalf, pfull_ref)
        np.testing.assert_allclose(result.values, [0.035467], rtol=2e-5)

    @pytest.mark.parametrize("increasing", [True, False])
    def test_closed_form(self, increasing: bool) -> None:
        """(p2 - p1) / ln(p2 / p1) for each layer, in either order.

        The log-mean also lies strictly between the two half levels and
        below their arithmetic mean. The second column is not a scaled copy of
        the first, which would give it the same log-pressure thicknesses.
        """
        column = np.array([100.0, 5000.0, 20000.0, 50000.0, 80000.0, 101325.0])
        phalf_vals = np.stack([column, 0.9 * column + 300.0])
        if not increasing:
            phalf_vals = phalf_vals[:, ::-1]
        phalf = xr.DataArray(
            phalf_vals, dims=["lat", PHALF_STR], coords={"lat": [0.0, 30.0]}
        )
        pfull_ref = xr.DataArray(
            np.arange(5.0), dims=[PFULL_STR], coords={PFULL_STR: np.arange(5.0)}
        )
        result = pfull_from_phalf_log_mean(phalf, pfull_ref)
        assert result.dims == ("lat", PFULL_STR)
        assert result.name == PFULL_STR
        np.testing.assert_array_equal(result[PFULL_STR].values, np.arange(5.0))
        p1, p2 = phalf_vals[:, :-1], phalf_vals[:, 1:]
        np.testing.assert_allclose(
            result.values, (p2 - p1) / np.log(p2 / p1), rtol=1e-14
        )
        assert (result.values > np.minimum(p1, p2)).all()
        assert (result.values < np.maximum(p1, p2)).all()
        assert (result.values < 0.5 * (p1 + p2)).all()

    @pytest.mark.parametrize("increasing", [True, False])
    @pytest.mark.parametrize(
        ("top", "message"), [(0.0, "pfull_simm_burr"), (-1.0, "negative")]
    )
    def test_nonpositive_half_level_raises(
        self, top: float, message: str, increasing: bool
    ) -> None:
        """A half level at or below zero pressure raises a ValueError.

        Only a zero-pressure top is sent to the other two methods; a negative
        pressure, such as from a fill value, is reported as invalid. With
        pressure decreasing along ``phalf``, the bad value is the last one.
        """
        vals = [top, 5000.0, 20000.0] if increasing else [20000.0, 5000.0, top]
        phalf = xr.DataArray(vals, dims=[PHALF_STR])
        pfull_ref = xr.DataArray([0.0, 1.0], dims=[PFULL_STR])
        with pytest.raises(ValueError, match=message):
            pfull_from_phalf_log_mean(phalf, pfull_ref)


@pytest.mark.parametrize("func", [pfull_from_phalf_avg, pfull_from_phalf_log_mean])
def test_pfull_from_phalf_dimension_names(func: Callable[..., xr.DataArray]) -> None:
    """Both means use the half- and full-level dimension names they are given."""
    column = np.array([100.0, 5000.0, 20000.0, 50000.0, 80000.0, 101325.0])
    phalf = xr.DataArray(
        np.stack([column, 0.9 * column + 300.0]),
        dims=["lat", PHALF_STR],
        coords={"lat": [0.0, 30.0]},
    )
    pfull_ref = xr.DataArray(np.arange(5.0), dims=[PFULL_STR])
    expected = func(phalf, pfull_ref)
    result = func(
        phalf.rename({PHALF_STR: "ph"}),
        pfull_ref.rename({PFULL_STR: "pf"}),
        phalf_str="ph",
        pfull_str="pf",
    )
    assert result.dims == ("lat", "pf")
    assert result.name == "pf"
    np.testing.assert_array_equal(result.values, expected.values)


# ---------------------------------------------------------------------------
# TestPfullValsSimBurr
# ---------------------------------------------------------------------------


class TestPfullValsSimBurr:
    """Tests for pfull_vals_simm_burr."""

    def _make_inputs(
        self, increasing: bool = True
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Create valid inputs for Simmons-Burridge calculation."""
        phalf_vals = np.array([100.0, 5000.0, 20000.0, 50000.0, 80000.0, 101325.0])
        pfull_vals = 0.5 * (phalf_vals[:-1] + phalf_vals[1:])
        if not increasing:
            phalf_vals = phalf_vals[::-1]
            pfull_vals = pfull_vals[::-1]
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

    @pytest.mark.parametrize("increasing", [True, False])
    def test_values_positive(self, increasing: bool) -> None:
        """All computed pressures are positive."""
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=increasing)
        result = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref)
        assert (result > 0).all()

    @pytest.mark.parametrize("increasing", [True, False])
    def test_values_between_half_levels(self, increasing: bool) -> None:
        """Full-level pressures lie between adjacent half-level pressures."""
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=increasing)
        result = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref)
        for i in range(len(result)):
            lo = min(phalf.values[i], phalf.values[i + 1])
            hi = max(phalf.values[i], phalf.values[i + 1])
            assert lo <= result[i] <= hi

    def test_order_invariance(self) -> None:
        """Reversing the vertical ordering reverses the output.

        Guards the decreasing-pressure branch: the top level is index -1 (not
        0) when pressure decreases with index, so the raw values must mirror
        the increasing case rather than mislabel the top and surface levels
        (issue #26).
        """
        phalf_i, phr_i, pfr_i = self._make_inputs(increasing=True)
        phalf_d, phr_d, pfr_d = self._make_inputs(increasing=False)
        inc = pfull_vals_simm_burr(phalf_i, phr_i, pfr_i)
        dec = pfull_vals_simm_burr(phalf_d, phr_d, pfr_d)
        np.testing.assert_allclose(dec, inc[::-1], rtol=1e-12)


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

    def test_returns_dataarray(self) -> None:
        """Return type is DataArray."""
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert isinstance(result, xr.DataArray)

    def test_increasing_pressure(self) -> None:
        """Works with increasing pressure ordering."""
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=True)
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert result.sizes[PFULL_STR] == pfull_ref.sizes[PFULL_STR]
        assert (result.values > 0).all()

    def test_decreasing_pressure(self) -> None:
        """Works with decreasing pressure ordering."""
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=False)
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert result.sizes[PFULL_STR] == pfull_ref.sizes[PFULL_STR]
        assert (result.values > 0).all()

    @pytest.mark.parametrize("increasing", [True, False])
    def test_matches_raw_numpy_values(self, increasing: bool) -> None:
        """The xarray wrapper reproduces the raw-numpy reference values.

        ``pfull_vals_simm_burr`` computes the Simmons-Burridge full-level
        pressures directly from numpy; ``pfull_simm_burr`` must return the
        same numbers on every column, for both pressure orderings (issue #26
        regression test).
        """
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=increasing)
        result = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        expected = pfull_vals_simm_burr(phalf.isel(time=0), phalf_ref, pfull_ref)
        for t in range(phalf.sizes["time"]):
            np.testing.assert_allclose(
                result.isel(time=t).transpose(PFULL_STR).values,
                expected,
                rtol=1e-12,
            )

    @pytest.mark.parametrize("increasing", [True, False])
    def test_labeled_pfull_coordinate(self, increasing: bool) -> None:
        """Works when ``pfull_ref`` carries a ``pfull`` dimension-coordinate.

        Selecting the top half level and ``expand_dims``-ing it yields an
        index-free length-1 ``pfull`` dimension. When ``pfull_not_top`` does
        carry a ``pfull`` index (as it does whenever ``pfull_ref`` is labeled),
        ``xr.concat`` used to raise "Dimension pfull already exists"; the top
        level must be labeled to match. Second half of the issue #26 fix.

        The check is against the coordinate-free result of the same call, so it
        isolates the coordinate handling from the pressure values themselves.
        """
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=increasing)
        labeled_ref = pfull_ref.assign_coords({PFULL_STR: pfull_ref.values})

        # Labeled input previously raised here; it must now succeed.
        result = pfull_simm_burr(phalf, phalf_ref, labeled_ref)
        baseline = pfull_simm_burr(phalf, phalf_ref, pfull_ref)

        # The pfull coordinate is preserved, in order, from pfull_ref.
        np.testing.assert_allclose(result[PFULL_STR].values, pfull_ref.values)
        # Labeling pfull_ref must not change the computed pressures.
        np.testing.assert_allclose(
            result.transpose(PFULL_STR, ...).values,
            baseline.transpose(PFULL_STR, ...).values,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("increasing", [True, False])
    def test_labeled_phalf_leaves_no_coordinate(self, increasing: bool) -> None:
        """A ``phalf`` coordinate on the input does not reach the output.

        The top level is computed from one half level, whose ``phalf`` value
        used to stay on the result as a scalar coordinate. That coordinate
        clashed with the ``phalf`` dimension of the Dataset in
        ``interp.interp_eta_to_plevs``.
        """
        phalf, phalf_ref, pfull_ref = self._make_inputs(increasing=increasing)
        labeled = phalf.assign_coords({PHALF_STR: phalf_ref.values})
        result = pfull_simm_burr(labeled, phalf_ref, pfull_ref)
        baseline = pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        assert PHALF_STR not in result.coords
        np.testing.assert_array_equal(
            result.transpose(PFULL_STR, ...).values,
            baseline.transpose(PFULL_STR, ...).values,
        )

    @pytest.mark.parametrize(
        ("bad_refs", "message"),
        [
            ("indices", "strictly between"),
            ("full levels in hPa", "strictly between"),
            ("full levels in hundredths of a Pa", "strictly between"),
            ("full levels on the half levels", "strictly between"),
            ("order", "opposite vertical orders"),
        ],
    )
    def test_references_must_be_pressures(self, bad_refs: str, message: str) -> None:
        """Index, mixed-unit or reversed references raise, in both functions.

        The top level is scaled by the ratio of the references, and the
        vertical order is read from them, so each of these would otherwise
        give a wrong top level, or invert the column, with no error.
        """
        phalf, phalf_ref, pfull_ref = self._make_inputs()
        n_half = phalf_ref.sizes[PHALF_STR]
        if bad_refs == "indices":
            phalf_ref = xr.DataArray(np.arange(n_half, dtype=float), dims=[PHALF_STR])
            pfull_ref = xr.DataArray(
                np.arange(n_half - 1, dtype=float), dims=[PFULL_STR]
            )
        elif bad_refs == "full levels in hPa":
            pfull_ref = pfull_ref / 100.0
        elif bad_refs == "full levels in hundredths of a Pa":
            pfull_ref = pfull_ref * 100.0
        elif bad_refs == "full levels on the half levels":
            pfull_ref = pfull_ref.copy(data=phalf_ref.values[1:])
        else:
            phalf_ref = phalf_ref.isel({PHALF_STR: slice(None, None, -1)})
            pfull_ref = pfull_ref.isel({PFULL_STR: slice(None, None, -1)})
        with pytest.raises(ValueError, match=message):
            pfull_simm_burr(phalf, phalf_ref, pfull_ref)
        with pytest.raises(ValueError, match=message):
            pfull_vals_simm_burr(phalf.isel(time=0), phalf_ref, pfull_ref)


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

    def test_decreasing_pressure_uniform_field(self) -> None:
        """Same constant average when phalf is ordered top-down (decreasing)."""
        phalf_vals = np.array([1e5, 8e4, 6e4, 4e4, 2e4, 1000.0])
        pfull_vals = 0.5 * (phalf_vals[:-1] + phalf_vals[1:])
        phalf = xr.DataArray(phalf_vals, dims=[PHALF_STR])
        pfull = xr.DataArray(pfull_vals, dims=[PFULL_STR])
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

    def test_decreasing_pressure_uniform_field(self) -> None:
        """Constant average is preserved under top-down (decreasing) ordering."""
        phalf_vals = np.array([1e5, 8e4, 6e4, 4e4, 2e4, 1000.0])
        pfull_vals = 0.5 * (phalf_vals[:-1] + phalf_vals[1:])
        phalf = xr.DataArray(phalf_vals, dims=[PHALF_STR])
        pfull = xr.DataArray(pfull_vals, dims=[PFULL_STR])
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

    def test_decreasing_pressure_detects_maximum(self) -> None:
        """Detection works when the pressure coord is decreasing (top-down)."""
        pfull = _pfull_decreasing()
        arr = xr.DataArray(
            [1.0, 3.0, 5.0, 3.0, 1.0],
            dims=[LEV_STR],
            coords={LEV_STR: pfull},
            name="peaked",
        )
        result = col_extrema(arr)
        assert not np.all(np.isnan(result.values))
