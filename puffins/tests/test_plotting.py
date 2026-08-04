"""Tests for the plotting module."""

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from puffins.plotting import (  # noqa: E402
    band_colors_about_center,
    panel_label,
    tight_levels,
    trunc_cmap_about_center,
    truncate_cmap,
)


@pytest.fixture
def ax() -> Iterator[matplotlib.axes.Axes]:
    """A bare Axes on a figure that is closed after the test."""
    fig, ax_ = plt.subplots()
    yield ax_
    plt.close(fig)


def _texts(ax_: matplotlib.axes.Axes) -> list[str]:
    """The strings of every Text artist added to the Axes."""
    return [t.get_text() for t in ax_.texts]


class TestPanelLabel:
    """Tests for panel_label."""

    def test_default_is_lowercase_in_parentheses(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """The historical default formatting is unchanged."""
        panel_label(0, ax=ax)
        assert _texts(ax) == ["(a)"]

    def test_counts_up_the_alphabet(self, ax: matplotlib.axes.Axes) -> None:
        """panel_num indexes the alphabet from zero."""
        panel_label(3, ax=ax)
        assert _texts(ax) == ["(d)"]

    def test_uppercase_gives_capital_letters(self, ax: matplotlib.axes.Axes) -> None:
        """uppercase=True switches to A, B, C..."""
        panel_label(1, ax=ax, uppercase=True)
        assert _texts(ax) == ["(B)"]

    def test_brackets_false_drops_the_parentheses(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """brackets=False leaves a bare letter."""
        panel_label(2, ax=ax, brackets=False)
        assert _texts(ax) == ["c"]

    def test_bold_sets_the_font_weight(self, ax: matplotlib.axes.Axes) -> None:
        """bold=True renders the label bold; the default does not."""
        panel_label(0, ax=ax, bold=True)
        assert ax.texts[0].get_fontweight() == "bold"

    def test_not_bold_by_default(self, ax: matplotlib.axes.Axes) -> None:
        """Without bold=True the weight is left at the rcParams default."""
        panel_label(0, ax=ax)
        assert ax.texts[0].get_fontweight() != "bold"

    def test_explicit_fontweight_wins_over_bold(self, ax: matplotlib.axes.Axes) -> None:
        """bold only supplies a default, so a passed fontweight takes priority."""
        panel_label(0, ax=ax, bold=True, fontweight="light")
        assert ax.texts[0].get_fontweight() == "light"

    def test_extra_text_is_appended(self, ax: matplotlib.axes.Axes) -> None:
        """extra_text follows the letter in the same Text artist."""
        panel_label(0, ax=ax, extra_text="JJAS")
        assert _texts(ax) == ["(a) JJAS"]

    def test_combined_options(self, ax: matplotlib.axes.Axes) -> None:
        """uppercase and brackets compose."""
        panel_label(0, ax=ax, uppercase=True, brackets=False, extra_text="rain")
        assert _texts(ax) == ["A rain"]

    def test_none_panel_num_labels_each_axes_in_turn(self) -> None:
        """Passing an iterable of Axes labels them a, b, c..."""
        fig, axarr = plt.subplots(1, 3)
        panel_label(ax=axarr)
        assert [_texts(a) for a in axarr] == [["(a)"], ["(b)"], ["(c)"]]
        plt.close(fig)

    def test_none_panel_num_propagates_the_style_options(self) -> None:
        """Style options reach each Axes in the iterable, not just the first."""
        fig, axarr = plt.subplots(1, 2)
        panel_label(ax=axarr, uppercase=True, brackets=False, bold=True)
        assert [_texts(a) for a in axarr] == [["A"], ["B"]]
        assert all(a.texts[0].get_fontweight() == "bold" for a in axarr)
        plt.close(fig)

    def test_rejects_panel_num_past_the_alphabet(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """Running off the end of the alphabet is an error, not an IndexError."""
        with pytest.raises(ValueError, match="panel_num must be between"):
            panel_label(26, ax=ax)

    @pytest.mark.parametrize("panel_num", [-1, -27])
    def test_rejects_negative_panel_num(
        self, panel_num: int, ax: matplotlib.axes.Axes
    ) -> None:
        """Negative indices are rejected rather than wrapping round the alphabet.

        Unguarded, -1 indexes the string from the back and silently labels the
        panel "(z)", while -27 raises the IndexError the check exists to
        replace.

        """
        with pytest.raises(ValueError, match="panel_num must be between"):
            panel_label(panel_num, ax=ax)

    def test_vertical_alignment_defaults_to_top(self, ax: matplotlib.axes.Axes) -> None:
        """The label anchors from its top edge unless told otherwise."""
        panel_label(0, ax=ax)
        assert ax.texts[0].get_verticalalignment() == "top"

    def test_vertical_alignment_can_be_overridden(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """An explicit va is honored, via either spelling."""
        panel_label(0, ax=ax, va="bottom")
        assert ax.texts[0].get_verticalalignment() == "bottom"


class TestTightLevels:
    """Levels bracket the data without leaving an unused end band."""

    def test_diverging_asymmetric(self) -> None:
        """Range asymmetric about zero brackets asymmetrically."""
        arr = np.array([-0.05433, 0.02864])
        levels = tight_levels(arr, 0.01)
        np.testing.assert_allclose(levels, np.arange(-6, 4) * 0.01, atol=1e-12)

    def test_end_bands_both_contain_data(self) -> None:
        """Neither the lowest nor the highest band is empty."""
        rng = np.random.default_rng(0)
        arr = rng.uniform(-3.904, 5.756, size=1000)
        arr[0], arr[1] = -3.904, 5.756
        step = 1.0
        levels = tight_levels(arr, step)
        assert levels[0] <= arr.min() < levels[0] + step
        assert levels[-1] - step < arr.max() <= levels[-1]

    def test_zero_is_a_level_when_data_span_zero(self) -> None:
        """Zero lands on a level, so a diverging cmap can center there."""
        levels = tight_levels(np.array([-3.9, 5.8]), 1.0)
        assert np.isclose(np.abs(levels).min(), 0.0)

    def test_data_flush_with_the_grid(self) -> None:
        """Data ending exactly on a multiple of step adds no extra band."""
        levels = tight_levels(np.array([0.0, 8.0]), 2.0)
        np.testing.assert_allclose(levels, [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_single_sign_data(self) -> None:
        """Wholly negative data bracket without reaching zero."""
        levels = tight_levels(np.array([-28.0, -6.0]), 5.0)
        np.testing.assert_allclose(levels, [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0])

    def test_nans_ignored(self) -> None:
        """NaNs do not propagate into the bracket."""
        arr = np.array([np.nan, 6.708, 28.88, np.nan])
        np.testing.assert_allclose(tight_levels(arr, 2.0), np.arange(6, 31, 2))

    def test_pooled_arrays(self) -> None:
        """A sequence of fields is pooled, as for a shared colorbar."""
        arrs = [
            np.array([-5.39, 7.592]),
            np.array([-3.066, 2.973]),
            np.array([-3.636, 3.389]),
        ]
        np.testing.assert_allclose(tight_levels(arrs, 1.0), np.arange(-6, 9))

    def test_dataarray_input(self) -> None:
        """xarray DataArrays work as well as numpy arrays."""
        arr = xr.DataArray([8.247, 52.16], dims=["x"])
        np.testing.assert_allclose(tight_levels(arr, 5.0), np.arange(5, 56, 5))

    def test_fractional_step_is_uniform(self) -> None:
        """A non-integer step accumulates no drift across the levels."""
        levels = tight_levels(np.array([0.0079, 3.352]), 0.4)
        np.testing.assert_allclose(np.diff(levels), 0.4)
        np.testing.assert_allclose(levels[-1], 3.6)

    @pytest.mark.parametrize("step", [0.0, -1.0])
    def test_nonpositive_step_raises(self, step: float) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            tight_levels(np.array([0.0, 1.0]), step)


class TestTruncCmapAboutCenter:
    """The center value stays at the colormap's midpoint color."""

    def test_asymmetric_range_keeps_zero_at_midpoint(self) -> None:
        """With a short positive side, zero still gets the midpoint color."""
        cmap = plt.get_cmap("RdBu_r")
        min_val, max_val = -0.06, 0.03
        trunc = trunc_cmap_about_center(cmap, min_val=min_val, max_val=max_val)
        frac_zero = (0.0 - min_val) / (max_val - min_val)
        np.testing.assert_allclose(trunc(frac_zero)[:3], cmap(0.5)[:3], atol=0.02)

    def test_short_side_stops_short_of_the_extreme(self) -> None:
        """The shorter side does not reach its end of the colormap."""
        cmap = plt.get_cmap("RdBu_r")
        trunc = trunc_cmap_about_center(cmap, min_val=-0.06, max_val=0.03)
        # Positive side spans half the negative side, so it should stop at the
        # 0.75 point of the original colormap rather than at 1.0.
        np.testing.assert_allclose(trunc(1.0)[:3], cmap(0.75)[:3], atol=0.02)
        np.testing.assert_allclose(trunc(0.0)[:3], cmap(0.0)[:3], atol=0.02)

    def test_symmetric_range_is_the_full_colormap(self) -> None:
        """A range symmetric about the center truncates nothing."""
        cmap = plt.get_cmap("RdBu_r")
        trunc = trunc_cmap_about_center(cmap, min_val=-5.0, max_val=5.0)
        for frac in (0.0, 0.5, 1.0):
            np.testing.assert_allclose(trunc(frac)[:3], cmap(frac)[:3], atol=0.02)

    def test_bounds_taken_from_array(self) -> None:
        """Passing arr instead of explicit bounds gives the same colormap."""
        cmap = plt.get_cmap("RdBu_r")
        arr = xr.DataArray([-0.06, 0.0, 0.03], dims=["x"])
        from_arr = trunc_cmap_about_center(cmap, arr=arr)
        explicit = trunc_cmap_about_center(cmap, min_val=-0.06, max_val=0.03)
        np.testing.assert_allclose(from_arr(0.4)[:3], explicit(0.4)[:3], atol=1e-10)


class TestBandColorsAboutCenter:
    """Per-band colors keep the central contour on the midpoint color."""

    @staticmethod
    def _positions(colors_out, cmap) -> np.ndarray:
        """Position in cmap whose color each band color matches."""
        grid = np.linspace(0.0, 1.0, 4001)
        ref = cmap(grid)[:, :3]
        return np.array(
            [
                grid[int(np.argmin(np.sum((ref - np.asarray(c)[:3]) ** 2, axis=1)))]
                for c in colors_out
            ]
        )

    def test_one_color_per_band(self) -> None:
        cmap = plt.get_cmap("RdBu_r")
        levels = np.arange(-6, 4) * 0.01
        assert len(band_colors_about_center(cmap, levels)) == len(levels) - 1

    def test_central_contour_on_midpoint_color(self) -> None:
        """The bands flanking zero sit equally either side of the midpoint."""
        cmap = plt.get_cmap("RdBu_r")
        levels = np.arange(-6, 4) * 0.01  # 6 bands below zero, 3 above
        pos = self._positions(band_colors_about_center(cmap, levels), cmap)
        ind_zero = int(np.argmin(np.abs(levels)))
        np.testing.assert_allclose(
            0.5 * (pos[ind_zero - 1] + pos[ind_zero]), 0.5, atol=0.005
        )

    def test_equal_color_increment_per_band(self) -> None:
        """Every band steps the same distance through the colormap."""
        cmap = plt.get_cmap("RdBu_r")
        pos = self._positions(
            band_colors_about_center(cmap, np.arange(-6, 4) * 0.01), cmap
        )
        np.testing.assert_allclose(np.diff(pos), np.diff(pos)[0], atol=0.005)

    def test_short_side_stops_short(self) -> None:
        """The side with fewer bands does not reach its end of the colormap."""
        cmap = plt.get_cmap("RdBu_r")
        pos = self._positions(
            band_colors_about_center(cmap, np.arange(-6, 4) * 0.01), cmap
        )
        assert pos[0] < 0.05  # long (negative) side nearly reaches its end
        assert 0.70 < pos[-1] < 0.75  # short side stops around 0.71

    def test_symmetric_levels_use_both_ends(self) -> None:
        cmap = plt.get_cmap("RdBu_r")
        pos = self._positions(
            band_colors_about_center(cmap, np.arange(-3, 4, 1.0)), cmap
        )
        np.testing.assert_allclose(pos, 1.0 - pos[::-1], atol=0.005)
        assert pos[0] < 0.1 and pos[-1] > 0.9

    def test_nonzero_central_val(self) -> None:
        """Centering on a value other than zero shifts which band is neutral."""
        cmap = plt.get_cmap("RdBu_r")
        levels = np.arange(280.0, 301.0, 2.0)
        pos = self._positions(
            band_colors_about_center(cmap, levels, central_val=290.0), cmap
        )
        ind_center = int(np.argmin(np.abs(levels - 290.0)))
        np.testing.assert_allclose(
            0.5 * (pos[ind_center - 1] + pos[ind_center]), 0.5, atol=0.005
        )

    def test_too_few_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 levels"):
            band_colors_about_center(plt.get_cmap("RdBu_r"), [0.0])


def test_truncate_cmap_endpoints() -> None:
    """Truncating maps the new endpoints onto the requested fractions."""
    cmap = plt.get_cmap("viridis")
    trunc = truncate_cmap(cmap, 0.25, 0.75)
    np.testing.assert_allclose(trunc(0.0)[:3], cmap(0.25)[:3], atol=0.02)
    np.testing.assert_allclose(trunc(1.0)[:3], cmap(0.75)[:3], atol=0.02)
