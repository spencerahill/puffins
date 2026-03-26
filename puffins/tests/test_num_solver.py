"""Tests for num_solver module."""

import numpy as np
import pytest
import xarray as xr

from puffins.num_solver import (
    brentq_solver_sweep_param,
    kj_from_n,
    n_from_kj,
    setup_bc_row,
    sor_solver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic_root_func(x: float, a: float) -> float:
    """f(x, a) = x^2 - a.  Root at x = sqrt(a) for positive a."""
    return x**2 - a


def _linear_func(x: float, a: float, slope: float) -> float:
    """f(x, a, slope) = slope * x - a.  Root at x = a / slope."""
    return slope * x - a


def _make_diag_dominant_system(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create diagonally dominant n×n system A*x = b with known solution x=1."""
    A = np.random.default_rng(42).standard_normal((n, n))
    # Make diagonally dominant for SOR convergence.
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1.0
    x_true = np.ones(n)
    b = A @ x_true
    return A, b, x_true


# ---------------------------------------------------------------------------
# brentq_solver_sweep_param
# ---------------------------------------------------------------------------


class TestBrentqSolverSweepParam:
    """Tests for brentq_solver_sweep_param."""

    def test_scalar_param_range(self) -> None:
        """Single scalar param_range finds the root."""
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=4.0,
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, [2.0], atol=1e-10)

    def test_int_scalar_param_range(self) -> None:
        """Integer scalar param_range also works."""
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=9,
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        np.testing.assert_allclose(result.values, [3.0], atol=1e-10)

    def test_list_param_range(self) -> None:
        """List of parameter values finds a root for each."""
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=[1.0, 4.0, 9.0],
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0], atol=1e-10)

    def test_ndarray_param_range(self) -> None:
        """numpy array param_range preserves shape."""
        params = np.array([1.0, 4.0, 16.0])
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=params,
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        np.testing.assert_allclose(result.values, [1.0, 2.0, 4.0], atol=1e-10)

    def test_dataarray_param_range_preserves_coords(self) -> None:
        """DataArray param_range preserves dims and coords."""
        params = xr.DataArray(
            [1.0, 4.0, 9.0],
            dims=["a_param"],
            coords={"a_param": [1.0, 4.0, 9.0]},
            name="params",
        )
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=params,
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        assert result.dims == ("a_param",)
        np.testing.assert_allclose(result.coords["a_param"].values, [1.0, 4.0, 9.0])
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0], atol=1e-10)

    def test_funcargs(self) -> None:
        """Extra funcargs are passed through to func."""
        result = brentq_solver_sweep_param(
            _linear_func,
            param_range=[3.0, 6.0],
            init_guess=0.0,
            bound_guess_range=np.arange(0.0, 20, 0.5),
            funcargs=(3.0,),  # slope=3 → root at a/3
        )
        np.testing.assert_allclose(result.values, [1.0, 2.0], atol=1e-10)

    def test_no_sign_change_returns_nan(self) -> None:
        """When no bracketing interval is found, result is NaN."""
        # x^2 - (-1) = x^2 + 1 has no real root; all guesses > 0.
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=-1.0,
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        assert np.isnan(result.values).item()

    def test_nan_in_middle_of_sweep(self) -> None:
        """NaN for unsolvable entry does not corrupt neighboring solutions."""
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=[4.0, -1.0, 9.0],
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        np.testing.assert_allclose(result.values[0], 2.0, atol=1e-10)
        assert np.isnan(result.values[1])
        np.testing.assert_allclose(result.values[2], 3.0, atol=1e-10)

    def test_returns_dataarray(self) -> None:
        """Return type is always xr.DataArray."""
        result = brentq_solver_sweep_param(
            _quadratic_root_func,
            param_range=4.0,
            init_guess=0.1,
            bound_guess_range=np.arange(0.1, 10, 0.5),
        )
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# sor_solver
# ---------------------------------------------------------------------------


class TestSorSolver:
    """Tests for sor_solver."""

    def test_identity_system(self) -> None:
        """Identity matrix: solution equals b."""
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x0 = np.zeros(3)
        result = sor_solver(A, b, x0, verbose=False)
        np.testing.assert_allclose(result, b, atol=1e-5)

    def test_diagonal_system(self) -> None:
        """Diagonal system with known solution."""
        A = np.diag([2.0, 4.0, 5.0])
        b = np.array([6.0, 8.0, 15.0])
        x0 = np.zeros(3)
        result = sor_solver(A, b, x0, verbose=False)
        np.testing.assert_allclose(result, [3.0, 2.0, 3.0], atol=1e-5)

    def test_diag_dominant_system(self) -> None:
        """Diagonally dominant system converges to known solution."""
        A, b, x_true = _make_diag_dominant_system(5)
        x0 = np.zeros(5)
        result = sor_solver(A, b, x0, omega=1.0, conv_crit=1e-8, verbose=False)
        np.testing.assert_allclose(result, x_true, atol=1e-5)

    def test_omega_one_is_gauss_seidel(self) -> None:
        """omega=1 reduces to Gauss-Seidel; still converges."""
        A, b, x_true = _make_diag_dominant_system(4)
        x0 = np.zeros(4)
        result = sor_solver(A, b, x0, omega=1.0, conv_crit=1e-8, verbose=False)
        np.testing.assert_allclose(result, x_true, atol=1e-5)

    def test_does_not_mutate_initial_guess(self) -> None:
        """The original initial_guess array is not modified."""
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x0 = np.zeros(3)
        x0_copy = x0.copy()
        sor_solver(A, b, x0, verbose=False)
        np.testing.assert_array_equal(x0, x0_copy)

    def test_verbose_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose=True prints residual lines."""
        A = np.eye(2)
        b = np.array([1.0, 2.0])
        x0 = np.zeros(2)
        sor_solver(A, b, x0, verbose=True)
        captured = capsys.readouterr()
        assert "Initial residual" in captured.out

    def test_initial_guess_already_exact(self) -> None:
        """When initial guess is already the solution, return immediately."""
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        result = sor_solver(A, b, b.copy(), verbose=False)
        np.testing.assert_allclose(result, b, atol=1e-10)

    def test_returns_ndarray(self) -> None:
        """Return type is np.ndarray."""
        A = np.eye(2)
        b = np.array([1.0, 1.0])
        result = sor_solver(A, b, np.zeros(2), verbose=False)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# n_from_kj / kj_from_n
# ---------------------------------------------------------------------------


class TestIndexConversion:
    """Tests for n_from_kj and kj_from_n."""

    def test_n_from_kj_simple(self) -> None:
        """Basic 2D-to-1D index conversion."""
        assert n_from_kj(0, 0, 5) == 0.0
        assert n_from_kj(0, 3, 5) == 3.0
        assert n_from_kj(2, 1, 5) == 11.0

    def test_kj_from_n_simple(self) -> None:
        """Basic 1D-to-2D index conversion."""
        assert kj_from_n(0, 5) == (0, 0)
        assert kj_from_n(3, 5) == (0, 3)
        assert kj_from_n(11, 5) == (2, 1)

    def test_round_trip(self) -> None:
        """Converting 2D→1D→2D recovers the original indices."""
        num_y = 7
        for k in range(4):
            for j in range(num_y):
                n = int(n_from_kj(k, j, num_y))
                k2, j2 = kj_from_n(n, num_y)
                assert (k2, j2) == (k, j)

    def test_n_from_kj_returns_numpy_floating(self) -> None:
        """Return type is np.floating (from np.rint)."""
        result = n_from_kj(1, 2, 3)
        assert isinstance(result, np.floating)

    def test_kj_from_n_returns_tuple(self) -> None:
        """Return type is a tuple of two ints."""
        result = kj_from_n(7, 3)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# setup_bc_row
# ---------------------------------------------------------------------------


class TestSetupBcRow:
    """Tests for setup_bc_row."""

    def test_sets_diagonal_to_one(self) -> None:
        """The diagonal entry of the BC row is set to 1."""
        matrix = np.zeros((4, 4))
        row = setup_bc_row(matrix, 1)
        assert row[1] == 1.0

    def test_sets_offdiag_to_bc(self) -> None:
        """Off-diagonal entries in the BC row are set to bc value."""
        matrix = np.ones((4, 4))
        row = setup_bc_row(matrix, 2, bc=0.0)
        assert row[0] == 0.0
        assert row[1] == 0.0
        assert row[3] == 0.0

    def test_custom_bc_value(self) -> None:
        """Non-zero bc value is applied to off-diagonal entries."""
        matrix = np.zeros((3, 3))
        row = setup_bc_row(matrix, 0, bc=5.0)
        assert row[0] == 1.0
        assert row[1] == 5.0
        assert row[2] == 5.0

    def test_modifies_matrix_in_place(self) -> None:
        """The original matrix is modified (row is a view)."""
        matrix = np.ones((3, 3))
        setup_bc_row(matrix, 1)
        np.testing.assert_array_equal(matrix[1], [0.0, 1.0, 0.0])

    def test_returns_row_view(self) -> None:
        """Returned array is the modified row of the matrix."""
        matrix = np.zeros((3, 3))
        row = setup_bc_row(matrix, 0)
        assert isinstance(row, np.ndarray)
        assert row.shape == (3,)
