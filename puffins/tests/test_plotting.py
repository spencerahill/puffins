"""Tests for the plotting module."""

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from puffins.plotting import (  # noqa: E402
    band_colors_about_center,
    colorbar_ticks,
    panel_label,
    sequential_band_colors,
    tight_levels,
    tight_scale,
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

    @pytest.mark.parametrize(
        ("low", "high", "step", "expected"),
        [
            (0.3, 0.7, 0.1, [0.3, 0.4, 0.5, 0.6, 0.7]),
            (0.7, 1.1, 0.1, [0.7, 0.8, 0.9, 1.0, 1.1]),
            (2.9, 3.1, 0.1, [2.9, 3.0, 3.1]),
            (0.03, 0.07, 0.01, [0.03, 0.04, 0.05, 0.06, 0.07]),
            (1.2, 2.4, 0.4, [1.2, 1.6, 2.0, 2.4]),
        ],
    )
    def test_data_flush_with_a_decimal_grid(
        self, low: float, high: float, step: float, expected: list[float]
    ) -> None:
        """An extremum exactly on a grid multiple adds no empty end band.

        ``low / step`` is an integer mathematically but not in binary floating
        point: 0.3 / 0.1 is 2.9999999999999996 and 0.07 / 0.01 is
        7.000000000000001, so a bare floor or ceil steps one band too far and
        produces exactly the empty end box this function exists to avoid.

        """
        levels = tight_levels(np.array([low, high]), step)
        np.testing.assert_allclose(levels, expected, atol=1e-12)

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

    def test_infinities_ignored(self) -> None:
        """Infinities drop out alongside NaNs rather than blowing up the grid."""
        arr = np.array([-np.inf, 6.708, 28.88, np.inf])
        np.testing.assert_allclose(tight_levels(arr, 2.0), np.arange(6, 31, 2))

    def test_pooled_arrays(self) -> None:
        """A sequence of fields is pooled, as for a shared colorbar.

        The minimum and the maximum live in different, non-leading arrays, so
        the assertion fails if any one field is dropped from the pool.

        """
        arrs = [
            np.array([-3.066, 2.973]),
            np.array([-5.39, 3.389]),
            np.array([-3.636, 7.592]),
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

    def test_constant_data_still_gives_a_band(self) -> None:
        """Constant data on a grid multiple yields two levels, not one.

        One level is zero bands, which ``contourf`` cannot use and which
        ``band_colors_about_center`` rejects outright.

        """
        levels = tight_levels(np.array([5.0, 5.0]), 1.0)
        np.testing.assert_allclose(levels, [5.0, 6.0])

    def test_all_nan_raises(self) -> None:
        """No finite values is a clear error, not a NaN-to-int crash."""
        with pytest.raises(ValueError, match="no finite values"):
            tight_levels(np.array([np.nan, np.nan]), 1.0)

    def test_empty_raises(self) -> None:
        """An empty field is the same clear error."""
        with pytest.raises(ValueError, match="no finite values"):
            tight_levels(np.array([]), 1.0)

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

    def test_decreasing_levels_raise(self) -> None:
        """Levels must increase; reversed ones would silently reverse the colors."""
        with pytest.raises(ValueError, match="must increase"):
            band_colors_about_center(plt.get_cmap("RdBu_r"), [3.0, 2.0, 1.0])

    def test_nan_levels_raise(self) -> None:
        """A NaN level would otherwise select a band silently and wrongly."""
        with pytest.raises(ValueError, match="must increase"):
            band_colors_about_center(plt.get_cmap("RdBu_r"), [0.0, np.nan, 1.0])

    def test_extend_neither_is_the_default(self) -> None:
        cmap = plt.get_cmap("RdBu_r")
        levels = np.arange(-8, 5) * 4.0
        assert band_colors_about_center(
            cmap, levels, extend="neither"
        ) == band_colors_about_center(cmap, levels)

    @pytest.mark.parametrize(
        ("levels", "under_pos", "over_pos"),
        [
            # Eight bands below zero and four above: the longer side's tip is
            # clipped to the colormap's end, and the shorter side's is one band
            # past its outermost band, at 0.5 + 4.5 / 16.
            (np.arange(-8, 5) * 4.0, 0.0, 0.78125),
            # Mirrored: one band below zero and eight above.
            (np.arange(-1, 9) * 4.0, 0.5 - 1.5 / 16, 1.0),
        ],
    )
    def test_tip_colors(
        self, levels: np.ndarray, under_pos: float, over_pos: float
    ) -> None:
        """Each tip is one band further along its side, clipped to the end."""
        cmap = plt.get_cmap("RdBu_r")
        bands = band_colors_about_center(cmap, levels)
        both = band_colors_about_center(cmap, levels, extend="both")
        assert both == [cmap(under_pos), *bands, cmap(over_pos)]
        assert band_colors_about_center(cmap, levels, extend="min") == both[:-1]
        assert band_colors_about_center(cmap, levels, extend="max") == both[1:]

    def test_tip_colors_reach_matplotlib_tips(self) -> None:
        """With BoundaryNorm, out-of-range values take the tip colors."""
        cmap = plt.get_cmap("RdBu_r")
        levels = np.arange(-8, 5) * 4.0
        colors = band_colors_about_center(cmap, levels, extend="both")
        listed = ListedColormap(colors)
        norm = BoundaryNorm(levels, len(colors), extend="both")
        assert listed(norm(-40.0)) == colors[0]
        assert listed(norm(17.0)) == colors[-1]
        assert listed(norm(1.0)) == colors[9]  # the band from 0 to 4

    def test_tip_colors_clipped_to_the_colormap(self) -> None:
        """A tip past the colormap's end takes its end color, not its own extremes.

        The colormap's under and over colors, black and green here, are what it
        returns for positions outside 0 to 1, so an unclipped position shows.

        """
        cmap = plt.get_cmap("RdBu_r").with_extremes(under="k", over="g")
        lower = band_colors_about_center(cmap, np.arange(-8, 5) * 4.0, extend="min")
        upper = band_colors_about_center(cmap, np.arange(-1, 9) * 4.0, extend="max")
        assert lower[0] == cmap(0.0) and upper[-1] == cmap(1.0)
        assert lower[0] != cmap(-0.1) and upper[-1] != cmap(1.1)

    def test_repeated_level_raises(self) -> None:
        with pytest.raises(ValueError, match="must increase"):
            band_colors_about_center(plt.get_cmap("RdBu_r"), [0.0, 0.0, 1.0])

    def test_bad_extend_raises(self) -> None:
        with pytest.raises(ValueError, match="extend must be one of"):
            band_colors_about_center(plt.get_cmap("RdBu_r"), [0.0, 1.0], extend="up")


class TestSequentialBandColors:
    """Bands sampled evenly from ``start``, stopping short of a tip."""

    def test_bands_span_start_to_end(self) -> None:
        cmap = plt.get_cmap("viridis")
        colors = sequential_band_colors(cmap, np.arange(9.0))
        assert colors == [cmap(p) for p in np.linspace(0.1, 1.0, 8)]

    def test_tip_beyond_every_band(self) -> None:
        """With a tip the last band sits at 1 - 0.8 / 8 and the tip at 1."""
        cmap = plt.get_cmap("viridis")
        colors = sequential_band_colors(cmap, np.arange(9.0), extend="max")
        assert colors == [*(cmap(p) for p in np.linspace(0.1, 0.9, 8)), cmap(1.0)]

    def test_start(self) -> None:
        cmap = plt.get_cmap("viridis")
        colors = sequential_band_colors(cmap, np.arange(5.0), start=0.3)
        assert colors == [cmap(p) for p in np.linspace(0.3, 1.0, 4)]

    def test_tip_reaches_matplotlib_tip(self) -> None:
        cmap = plt.get_cmap("viridis")
        levels = np.arange(9.0)
        colors = sequential_band_colors(cmap, levels, extend="max")
        norm = BoundaryNorm(levels, len(colors), extend="max")
        assert ListedColormap(colors)(norm(20.0)) == cmap(1.0)

    @pytest.mark.parametrize("extend", ["min", "both"])
    def test_lower_tip_raises(self, extend: str) -> None:
        with pytest.raises(ValueError, match="tips only at its upper end"):
            sequential_band_colors(plt.get_cmap("viridis"), [0.0, 1.0], extend=extend)

    @pytest.mark.parametrize(
        ("start", "extend"), [(-0.1, "neither"), (0.95, "max"), (0.9, "max")]
    )
    def test_start_out_of_range_raises(self, start: float, extend: str) -> None:
        """The last two lie at or past the last band, at 0.9 for eight bands."""
        with pytest.raises(ValueError, match="start must lie"):
            sequential_band_colors(
                plt.get_cmap("viridis"), np.arange(9.0), start=start, extend=extend
            )

    def test_start_at_zero(self) -> None:
        cmap = plt.get_cmap("viridis")
        colors = sequential_band_colors(cmap, np.arange(5.0), start=0.0)
        assert colors[0] == cmap(0.0)


class TestTightScale:
    """Levels per Spencer Hill's colorbar rules of 2026-09-28.

    The expected levels are worked out by hand in each test's docstring from
    the rules in ``tight_scale``'s docstring.  "Nice" band widths are the
    mantissas 1, 1.5, 2, 2.5, 3, 4, 5, 6 and 8 times a power of ten for a
    diverging bar, and 1, 2, 2.5 and 5 for a sequential one.

    """

    def test_diverging_asymmetric(self) -> None:
        """Eight bands on the longer side, as many as reach the shorter's extreme.

        The longer, negative side covers 31.1: the smallest nice width whose
        eight bands reach it is 4, and its outermost band, -32 to -28, holds
        -31.1.  The positive side needs ceil(14 / 4) = 4 bands.

        """
        scale = tight_scale(np.array([-31.1, 14.0]))
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 5) * 4.0)
        assert scale.extend == "neither"
        assert scale.step == 4.0
        assert scale.extremes == (-31.1, 14.0)

    def test_outermost_band_never_empty(self) -> None:
        """A width rounded up past one band beyond the extreme is refused.

        Eight bands of 4 reach -24.5 but leave -32 to -28 empty, since -24.5 is
        above -28.  So the width is the largest nice one short of 24.5, which
        is 3, and the last 0.5 runs into a tip.

        """
        scale = tight_scale(np.array([-24.5, 10.0]))
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 5) * 3.0)
        assert scale.extend == "min"

    def test_long_side_tips_past_its_reference(self) -> None:
        """An extreme beyond 1.3 times the 95th percentile runs into a tip.

        The negative side's 95th percentile of magnitude lies between 16 and
        20, and its extreme, 60, is far past 1.3 times that, so its eight bands
        cover the percentile at a width of 2.5.  The positive side reaches 5
        in two bands.

        """
        field = np.concatenate([np.linspace(-20.0, 5.0, 2501), [-60.0]])
        reference = np.quantile(-field[field < 0], 0.95)
        assert 16.0 < reference <= 20.0 and 1.3 * reference < 60.0
        scale = tight_scale(field)
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 3) * 2.5)
        assert scale.extend == "min"
        assert scale.extremes == (-60.0, 5.0)

    def test_shorter_side_never_tips_so_bar_turns_symmetric(self) -> None:
        """A shorter side needing as many bands as the longer gets exactly as many.

        The negative side covers 10.  Eight bands of 1.5 would leave -12 to
        -10.5 empty, so the width is 1.  The positive side would then need ten
        bands to reach 9.9; the bar becomes symmetric at eight a side, and
        both sides tip.

        """
        scale = tight_scale(np.array([-10.0, 9.9]))
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 9) * 1.0)
        assert scale.extend == "both"

    def test_more_than_ten_bands_per_side(self) -> None:
        """Twenty bands on the longer side, more than one decade below its cover.

        The negative side covers 1.0, so its width is 1 / 20 = 0.05, a nice
        width; the positive side reaches 0.5 in ten bands.

        """
        scale = tight_scale(np.array([-1.0, 0.5]), bands_per_side=20)
        assert scale.step == 0.05
        np.testing.assert_allclose(
            scale.levels, np.arange(-20, 11) * 0.05, rtol=0, atol=1e-12
        )
        assert scale.extend == "neither"

    def test_shorter_side_end_on_a_grid_multiple(self) -> None:
        """0.07 / 0.01 is 7.000000000000001, which must not add an empty band."""
        scale = tight_scale(np.array([-0.08, 0.07]), step=0.01)
        assert scale.levels[-1] == 0.07
        assert scale.levels[0] == -0.08
        assert scale.extend == "neither"

    def test_levels_rounded(self) -> None:
        """No level carries float noise, as -6 times 0.006, -0.036000000000000004, does.

        The negative side covers 0.0543 at a width of 0.006, eight bands to
        -0.048, since a width of 0.008 would leave -0.064 to -0.056 empty; the
        positive side reaches 0.0286 in five bands, to 0.03.

        """
        scale = tight_scale(np.array([-0.0543, 0.0286]))
        expected = [float(f"{v:.12g}") for v in np.arange(-8, 6) * 0.006]
        assert scale.levels.tolist() == expected
        assert [str(v) for v in scale.levels[:3]] == ["-0.048", "-0.042", "-0.036"]
        assert scale.extend == "min"

    def test_step(self) -> None:
        """A given width covers each side's extreme in as many bands as needed."""
        scale = tight_scale(np.array([-31.1, 14.0]), step=5.0)
        np.testing.assert_array_equal(scale.levels, np.arange(-7, 4) * 5.0)
        assert scale.step == 5.0

    def test_step_with_a_tip(self) -> None:
        """A given width covers a tipping side's reference, not its extreme.

        The field of ``test_long_side_tips_past_its_reference``: four bands of
        5 cover its negative side's reference, between 16 and 20, and one
        reaches 5.

        """
        field = np.concatenate([np.linspace(-20.0, 5.0, 2501), [-60.0]])
        scale = tight_scale(field, step=5.0)
        np.testing.assert_array_equal(scale.levels, np.arange(-4, 2) * 5.0)
        assert scale.extend == "min"

    def test_bands_per_side(self) -> None:
        """Four bands of 8 cover -31.1, and two reach 14."""
        scale = tight_scale(np.array([-31.1, 14.0]), bands_per_side=4)
        np.testing.assert_array_equal(scale.levels, np.arange(-4, 3) * 8.0)

    def test_sequential(self) -> None:
        """Up to sixteen bands from the smallest value to the largest.

        The range of June-September mean rain over monsoonal India, 0.3683 to
        39.2353 mm per day: 2.5 is the finest sequential width giving at most
        sixteen bands, 0 to 40.

        """
        scale = tight_scale(np.array([0.3683, 39.2353]), kind="sequential")
        np.testing.assert_array_equal(scale.levels, np.arange(17) * 2.5)
        assert scale.extend == "neither"
        assert scale.step == 2.5

    def test_sequential_num_bands(self) -> None:
        """At most eight bands: a width of 5, 0 to 40."""
        scale = tight_scale(np.array([0.3683, 39.2353]), kind="sequential", num_bands=8)
        np.testing.assert_array_equal(scale.levels, np.arange(9) * 5.0)

    def test_sequential_far_from_zero(self) -> None:
        """The bar starts at the smallest value: 150 to 300 in fifteen bands of 10."""
        scale = tight_scale(np.array([151.2, 298.7]), kind="sequential")
        np.testing.assert_array_equal(scale.levels, 150.0 + np.arange(16) * 10.0)

    def test_sequential_top_tips(self) -> None:
        """The bands cover a 95th percentile between 9 and 10, in ten bands of 1."""
        field = np.concatenate([np.linspace(0.0, 10.0, 1001), [40.0]])
        reference = np.quantile(field, 0.95)
        assert 9.0 < reference <= 10.0 and 1.3 * reference < 40.0
        scale = tight_scale(field, kind="sequential")
        np.testing.assert_array_equal(scale.levels, np.arange(11) * 1.0)
        assert scale.extend == "max"

    def test_sequential_mostly_at_the_minimum(self) -> None:
        """A 95th percentile at the minimum leaves the bands covering the extreme.

        Like a rain field that is zero almost everywhere: fourteen bands of 0.5
        reach 7.

        """
        field = np.concatenate([np.zeros(1000), [5.0, 7.0]])
        assert np.quantile(field, 0.95) == 0.0
        scale = tight_scale(field, kind="sequential")
        np.testing.assert_array_equal(scale.levels, np.arange(15) * 0.5)
        assert scale.extend == "neither"

    def test_reference_is_per_field_not_pooled(self) -> None:
        """A field of small values sharing the bar does not pull the reference down.

        The first field's own 95th percentile is about 9.5, and its extreme,
        12, is within 1.3 times that, so the bar covers 12.  The second field
        alone lies between 0 and 1, and the 95th percentile of both pooled is
        low enough that the pooled rule would tip at 12.

        """
        first = np.concatenate([np.linspace(0.0, 10.0, 1001), [12.0]])
        second = np.tile(np.linspace(0.0, 1.0, 101), 30)
        assert 1.3 * np.quantile(first, 0.95) >= 12.0
        assert 1.3 * np.quantile(np.concatenate([first, second]), 0.95) < 12.0
        scale = tight_scale([first, second], kind="sequential")
        np.testing.assert_array_equal(scale.levels, np.arange(13) * 1.0)
        assert scale.extend == "neither"

    def test_reference_takes_every_field(self) -> None:
        """The extremes and the reference come from every field, in any position."""
        fields = (
            xr.DataArray(np.linspace(0.0, 1.0, 101), dims=["x"]),
            xr.DataArray(np.concatenate([np.linspace(0.0, 10.0, 1001), [12.0]])),
        )
        scale = tight_scale(fields, kind="sequential")
        assert scale.extremes == (0.0, 12.0)
        assert scale.extend == "neither"
        np.testing.assert_array_equal(scale.levels, np.arange(13) * 1.0)

    def test_quantile(self) -> None:
        """At the 90th percentile, 9, the extreme of 12 passes 1.3 times it."""
        field = np.concatenate([np.linspace(0.0, 10.0, 1001), [12.0]])
        assert tight_scale(field, kind="sequential").extend == "neither"
        assert tight_scale(field, kind="sequential", quantile=0.9).extend == "max"

    def test_tip_ratio(self) -> None:
        """At a ratio of 1.1 the extreme of 12 passes 1.1 times about 9.5."""
        field = np.concatenate([np.linspace(0.0, 10.0, 1001), [12.0]])
        assert tight_scale(field, kind="sequential", tip_ratio=1.1).extend == "max"

    def test_nans_and_infinities_ignored(self) -> None:
        field = np.array([np.nan, -31.1, np.inf, 14.0, -np.inf])
        np.testing.assert_array_equal(tight_scale(field).levels, np.arange(-8, 5) * 4.0)

    def test_masked_cells_ignored(self) -> None:
        """A masked array's masked cells, here holding 1e20, are left out."""
        field = np.linspace(-3.0, 2.0, 1000)
        masked = np.ma.masked_array(field, mask=np.arange(1000) % 3 == 0)
        masked.data[masked.mask] = 1e20
        want = tight_scale(np.where(masked.mask, np.nan, field))
        got = tight_scale(masked)
        np.testing.assert_array_equal(got.levels, want.levels)
        assert got.extremes == want.extremes and got.extend == want.extend

    def test_list_of_numbers_is_one_field(self) -> None:
        """Read as one field per number, no side could tip; read as one, it does."""
        field = np.concatenate([np.linspace(-20.0, 5.0, 2501), [-60.0]])
        got = tight_scale(field.tolist())
        np.testing.assert_array_equal(got.levels, tight_scale(field).levels)
        assert got.extend == "min"

    @pytest.mark.parametrize(
        ("field", "kind", "step", "end", "value"),
        [
            # 0.3 - 0.1 is 0.19999999999999998, a hair below the level at 0.2.
            ([0.3 - 0.1, 3.0], "sequential", None, 0, 0.3 - 0.1),
            # 0.1 + 0.2 is 0.30000000000000004, a hair above the level at 0.3.
            ([0.0, 0.1 + 0.2], "sequential", None, -1, 0.1 + 0.2),
            # 0.8 * 3 is 2.4000000000000004, a hair past the level at -2.4.
            ([-(0.8 * 3), 1.0], "diverging", None, 0, -(0.8 * 3)),
            # The shorter side's end, 0.1 * 3, a hair above the level at 0.3.
            ([-0.8, 0.1 * 3], "diverging", 0.1, -1, 0.1 * 3),
        ],
    )
    def test_extreme_a_hair_past_a_level_is_snapped(
        self, field: list[float], kind: str, step: float | None, end: int, value: float
    ) -> None:
        """An extreme past an end level by rounding alone moves the level, not a tip."""
        scale = tight_scale(np.array(field), kind=kind, step=step)
        assert scale.extend == "neither"
        assert scale.levels[end] == value

    def test_snapped_sequential_bar_takes_its_colors(self) -> None:
        """The snapped bar's extend is one sequential_band_colors accepts."""
        scale = tight_scale(np.array([0.3 - 0.1, 3.0]), kind="sequential")
        colors = sequential_band_colors(
            plt.get_cmap("viridis"), scale.levels, extend=scale.extend
        )
        assert len(colors) == scale.levels.size - 1

    def test_shorter_side_snapped_adds_no_band(self) -> None:
        """-0.8 to 0.1 * 3 in bands of 0.1 is eleven bands, not twelve."""
        scale = tight_scale(np.array([-0.8, 0.1 * 3]), step=0.1)
        assert scale.levels.size == 12

    def test_longer_side_on_a_grid_multiple(self) -> None:
        """0.07 / 0.01 is 7.000000000000001, which must not add a band below -0.07."""
        scale = tight_scale(np.array([-0.07, 0.01]), step=0.01)
        assert scale.levels[0] == -0.07 and scale.levels.size == 9

    def test_tiny_range_far_from_zero(self) -> None:
        """Rounding relative to the band width keeps levels a millionth apart distinct."""
        scale = tight_scale(np.array([1e6, 1e6 + 1e-6]), kind="sequential")
        assert scale.levels.size >= 2 and np.all(np.diff(scale.levels) > 0)

    def test_sequential_offset_from_zero(self) -> None:
        """10000 to 10010 in ten bands of 1, the width set by the range, not 10010."""
        scale = tight_scale(np.array([10000.0, 10010.0]), kind="sequential")
        np.testing.assert_array_equal(scale.levels, 10000.0 + np.arange(11))

    def test_sequential_one_band(self) -> None:
        """4.95 to 5.05 in at most one band: no width up to 1 does it, 2 does."""
        scale = tight_scale(np.array([4.95, 5.05]), kind="sequential", num_bands=1)
        np.testing.assert_array_equal(scale.levels, [4.0, 6.0])

    def test_sequential_noise_below_zero(self) -> None:
        """A minimum of -1e-15, regridding noise, is zero to within rounding."""
        scale = tight_scale(np.array([-1e-15, 3.0]), kind="sequential")
        assert scale.levels[0] == -1e-15 and scale.extend == "neither"

    def test_sequential_reference_counts_every_value(self) -> None:
        """The 95th percentile of 80 zeros, 1 to 10 and 12 is about 8.1, zeros included.

        So 12 passes 1.3 times it and the bar covers it in nine bands of 1.  Left
        out, the zeros would raise the percentile to about 10, and 12 would not
        tip.

        """
        field = np.concatenate([np.zeros(80), np.linspace(1.0, 10.0, 20), [12.0]])
        scale = tight_scale(field, kind="sequential")
        np.testing.assert_array_equal(scale.levels, np.arange(10) * 1.0)
        assert scale.extend == "max"

    def test_ratio_exactly_met_does_not_tip(self) -> None:
        """An extreme of exactly 2 times a median of 1, at a ratio of 2, is covered."""
        scale = tight_scale(
            np.array([0.0, 1.0, 2.0]), kind="sequential", quantile=0.5, tip_ratio=2.0
        )
        np.testing.assert_allclose(scale.levels, np.arange(11) * 0.2, atol=1e-12)
        assert scale.extend == "neither"

    def test_zeros_are_on_neither_side(self) -> None:
        """Zeros do not count toward the positive side's percentile.

        The positive values 1, 1, 1 and 1.2 have a 95th percentile of 1.17,
        which 1.2 is within 1.3 times of, so eight bands of 0.15 cover 1.2;
        counted in, the field's thousand zeros would make it tip.

        """
        field = np.concatenate([np.zeros(1000), [1.0, 1.0, 1.0, 1.2, -0.5]])
        scale = tight_scale(field)
        np.testing.assert_allclose(scale.levels, np.arange(-4, 9) * 0.15, atol=1e-12)
        assert scale.extend == "neither"

    def test_field_with_one_side_empty(self) -> None:
        """A field with no values on one side of zero, beside one that has them.

        The negative side covers 3 in eight bands of 0.4, to -3.2, and the
        positive side reaches 2 in five.

        """
        scale = tight_scale([np.array([-3.0, -1.0]), np.array([1.0, 2.0])])
        np.testing.assert_allclose(scale.levels, np.arange(-8, 6) * 0.4, atol=1e-12)
        assert scale.extend == "neither"

    def test_field_with_no_finite_values_beside_others(self) -> None:
        scale = tight_scale([np.array([-3.0, 2.0]), np.array([np.nan])])
        want = tight_scale(np.array([-3.0, 2.0]))
        np.testing.assert_array_equal(scale.levels, want.levels)

    @pytest.mark.parametrize(
        ("field", "levels", "extend"),
        [
            ([-10.0, -2.0], np.arange(-8, 1) * 1.0, "min"),
            ([2.0, 10.0], np.arange(0, 9) * 1.0, "max"),
        ],
    )
    def test_one_signed_diverging_bar_starts_at_zero(
        self, field: list[float], levels: np.ndarray, extend: str
    ) -> None:
        """A field of one sign gets bands from zero, eight of width 1 to cover 10."""
        scale = tight_scale(np.array(field))
        np.testing.assert_array_equal(scale.levels, levels)
        assert scale.extend == extend

    def test_quantile_on_a_diverging_bar(self) -> None:
        """At the median, the negative reference is about 10 and the positive 2.5.

        The negative side covers 10.01 in eight bands of 1.5, and the positive
        side, as the shorter, reaches its extreme of 5 in four bands.

        """
        field = np.concatenate([np.linspace(-20.0, 5.0, 2501), [-60.0]])
        scale = tight_scale(field, quantile=0.5)
        np.testing.assert_allclose(scale.levels, np.arange(-8, 5) * 1.5, atol=1e-12)
        assert scale.extend == "min"

    def test_tip_ratio_on_a_diverging_bar(self) -> None:
        """At a ratio of 5, neither side tips: eight bands of 8 cover 60."""
        field = np.concatenate([np.linspace(-20.0, 5.0, 2501), [-60.0]])
        scale = tight_scale(field, tip_ratio=5.0)
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 2) * 8.0)
        assert scale.extend == "neither"

    def test_covers_tied_make_the_positive_side_longer(self) -> None:
        """Both sides cover 10 at the median; the positive is taken as longer.

        Eight bands of 1 then go to 8, and the negative side, needing a hundred
        to reach -100, is capped at eight, so the bar is symmetric and tips at
        both ends.

        """
        scale = tight_scale(np.array([-100.0, -10.0, -1.0, 10.0]), quantile=0.5)
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 9) * 1.0)
        assert scale.extend == "both"

    def test_tipping_side_judged_on_its_extreme(self) -> None:
        """Eight bands of 2.5 cover a reference of 17; the extreme of 60 fills the last.

        Judged on the reference instead, eight bands of 2.5 would leave 17.5 to
        20 empty, and the width would drop to 2.

        """
        field = np.concatenate([np.full(99, -17.0), [-60.0, 1.0]])
        scale = tight_scale(field)
        assert scale.step == 2.5
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 2) * 2.5)

    def test_extreme_at_seven_widths_takes_the_narrower(self) -> None:
        """-28 does not pass seven bands of 4, so the width drops to 3 and -28 tips."""
        scale = tight_scale(np.array([-28.0, 1.0]))
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 2) * 3.0)
        assert scale.extend == "min"

    def test_bands_per_side_not_a_power_of_two(self) -> None:
        """Six bands of 0.3 reach 1.8 exactly, which a looser test would refuse."""
        field = np.concatenate([np.full(99, -1.8), [-5.0, 0.5]])
        scale = tight_scale(field, bands_per_side=6)
        assert scale.step == 0.3

    @pytest.mark.parametrize("seed", range(20))
    def test_rules_hold_on_random_fields(self, seed: int) -> None:
        """The rules as invariants, over shifted and heavy-tailed random fields.

        The longer side has exactly eight bands; an asymmetric bar's shorter
        side never tips and its outermost band holds data; and the longer
        side's outermost band holds data unless that side tips.  The shift of
        up to 3 standard deviations makes some fields one-signed.

        """
        rng = np.random.default_rng(seed)
        fields = []
        for _ in range(rng.integers(1, 4)):
            vals = 10.0 ** rng.uniform(-3, 3) * (
                rng.standard_normal(int(rng.integers(20, 2000))) + rng.uniform(-3, 3)
            )
            vals[:5] *= rng.uniform(1, 8)
            fields.append(vals)
        scale = tight_scale(fields)
        levels, step = scale.levels, scale.step
        lo, hi = scale.extremes
        n_below = int(np.sum(levels < 0))
        n_above = int(np.sum(levels > 0))
        assert max(n_below, n_above) == 8
        tips_low, tips_high = lo < levels[0], hi > levels[-1]
        assert (
            scale.extend
            == {
                (False, False): "neither",
                (True, False): "min",
                (False, True): "max",
                (True, True): "both",
            }[(tips_low, tips_high)]
        )
        if n_below > n_above:
            assert not tips_high
            assert tips_low or -lo > (n_below - 1) * step
            assert n_above == 0 or hi > (n_above - 1) * step
        elif n_above > n_below:
            assert not tips_low
            assert tips_high or hi > (n_above - 1) * step
            assert n_below == 0 or -lo > (n_below - 1) * step

    def test_constant_with_step(self) -> None:
        scale = tight_scale(np.array([5.0, 5.0]), kind="sequential", step=1.0)
        np.testing.assert_array_equal(scale.levels, [5.0, 6.0])

    def test_constant_zero_with_step(self) -> None:
        scale = tight_scale(np.zeros(3), step=1.0)
        np.testing.assert_array_equal(scale.levels, [0.0, 1.0])

    @pytest.mark.parametrize(
        ("value", "kind"), [(0.0, "diverging"), (5.0, "diverging"), (5.0, "sequential")]
    )
    def test_constant_without_step_raises(self, value: float, kind: str) -> None:
        with pytest.raises(ValueError, match="single value"):
            tight_scale(np.full(3, value), kind=kind)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bands_per_side": 0}, "at least 1"),
            ({"num_bands": 0}, "at least 1"),
            ({"quantile": 0.0}, "quantile must"),
            ({"quantile": 1.5}, "quantile must"),
            ({"tip_ratio": 0.0}, "tip_ratio must"),
            ({"step": np.nan}, "step must be positive"),
        ],
    )
    def test_bad_parameters_raise(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            tight_scale(np.array([-1.0, 1.0]), **kwargs)  # type: ignore[arg-type]

    def test_quantile_of_one_is_allowed(self) -> None:
        """At quantile 1 the reference is the extreme itself, so nothing tips."""
        field = np.concatenate([np.linspace(-20.0, 5.0, 2501), [-60.0]])
        assert tight_scale(field, quantile=1.0).extend == "neither"

    def test_negative_sequential_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            tight_scale(np.array([-1.0, 5.0]), kind="sequential")

    def test_bad_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            tight_scale(np.array([0.0, 1.0]), kind="cyclic")

    def test_no_finite_values_raises(self) -> None:
        with pytest.raises(ValueError, match="no finite values"):
            tight_scale([np.array([np.nan]), np.array([])])

    @pytest.mark.parametrize("step", [0.0, -1.0])
    def test_nonpositive_step_raises(self, step: float) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            tight_scale(np.array([-1.0, 1.0]), step=step)


class TestColorbarTicks:
    """Labeled boundaries: outward from zero, or at a stride dividing the bar."""

    def test_outward_from_zero(self) -> None:
        levels = np.arange(-8, 5) * 4.0
        assert colorbar_ticks(levels) == [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0]

    def test_zero_labeled_when_off_the_ends_stride(self) -> None:
        """Three bands below zero and nine above: the ends go unlabeled."""
        levels = np.arange(-3, 10) * 1.0
        assert colorbar_ticks(levels) == [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

    def test_every(self) -> None:
        levels = np.arange(-8, 9) * 1.0
        assert colorbar_ticks(levels, every=4) == [-8.0, -4.0, 0.0, 4.0, 8.0]

    def test_coarsens_to_max_labels(self) -> None:
        """Every second boundary gives nine labels; at most five takes every third."""
        levels = np.arange(-8, 9) * 1.0
        assert len(colorbar_ticks(levels)) == 9
        assert colorbar_ticks(levels, max_labels=5) == [-6.0, -3.0, 0.0, 3.0, 6.0]

    def test_sequential_stride_divides_the_bar(self) -> None:
        levels = np.arange(17) * 2.5
        assert colorbar_ticks(levels) == [
            0.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            35.0,
            40.0,
        ]

    def test_prefers_four_or_more_labels(self) -> None:
        """Twelve bands ending at zero, every=6: four labels beat three.

        Strides of 4 and 6 both divide the bar; 6 is nearer ``every`` but
        gives only three labels.

        """
        levels = np.arange(-12, 1) * 1.0
        assert colorbar_ticks(levels, every=6) == [-12.0, -8.0, -4.0, 0.0]

    def test_tie_goes_to_the_coarser_stride(self) -> None:
        """Sixteen bands, every=3: strides 2 and 4 are equally near; 4 wins."""
        levels = np.arange(17) * 1.0
        assert colorbar_ticks(levels, every=3) == [0.0, 4.0, 8.0, 12.0, 16.0]

    def test_prime_band_count_without_zero(self) -> None:
        """Thirteen bands: no divisor but 1 and 13, so count from the lowest."""
        levels = np.arange(1, 15) * 1.0
        assert colorbar_ticks(levels) == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]

    def test_prime_band_count_from_zero(self) -> None:
        levels = np.arange(-13, 1) * 1.0
        assert colorbar_ticks(levels) == [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0]

    def test_near_zero_level_labeled_as_zero(self) -> None:
        """-0.3 + 3 * 0.1 is 5.55e-17, which must not print as such."""
        levels = -0.3 + 0.1 * np.arange(7)
        assert levels[3] != 0.0
        ticks = colorbar_ticks(levels)
        assert 0.0 in ticks
        assert not any(0.0 < abs(t) < 1e-12 for t in ticks)

    def test_zero_label_has_no_sign(self) -> None:
        """A level a hair below zero is labeled 0, not -0, which prints as -0."""
        ticks = colorbar_ticks([-2.0, -1.0, -1e-12, 1.0, 2.0])
        assert ticks == [-2.0, 0.0, 2.0]
        assert not np.signbit(ticks[1])

    def test_large_levels_rounded(self) -> None:
        """Levels a part in 1e13 off multiples of 1500 label as the multiples.

        Twelve decimal places would keep the noise; six significant figures
        of the band width do not.

        """
        noise = np.array([3e-10, -2e-10, 1e-10, 4.6e-10, -1e-10, 3e-10])
        levels = np.arange(-3, 3) * 1500.0 + noise
        assert colorbar_ticks(levels, every=1) == [
            -4500.0,
            -3000.0,
            -1500.0,
            0.0,
            1500.0,
            3000.0,
        ]

    def test_bad_every_raises(self) -> None:
        with pytest.raises(ValueError, match="every must be"):
            colorbar_ticks(np.arange(5.0), every=0)

    def test_two_levels(self) -> None:
        assert colorbar_ticks([0.0, 1.0]) == [0.0, 1.0]

    @pytest.mark.parametrize(
        ("levels", "every", "max_labels", "want"),
        [
            # Six bands, no zero: strides 2 and 3 divide; 3 is nearest every=3.
            (np.arange(1.0, 8.0), 3, 9, [1.0, 4.0, 7.0]),
            # Nine bands: stride 1 gives ten labels and 9 gives two, so 3.
            (np.arange(1.0, 11.0), 1, 9, [1.0, 4.0, 7.0, 10.0]),
            # Sixteen bands from zero, at most five labels: stride 4.
            (np.arange(17.0), 2, 5, [0.0, 4.0, 8.0, 12.0, 16.0]),
        ],
    )
    def test_divisor_path(
        self, levels: np.ndarray, every: int, max_labels: int, want: list[float]
    ) -> None:
        assert colorbar_ticks(levels, every=every, max_labels=max_labels) == want

    @pytest.mark.parametrize(
        ("every", "max_labels", "want"),
        [
            (10, 9, [1.0, 7.0, 13.0]),
            (2, 5, [1.0, 4.0, 7.0, 10.0, 13.0]),
            (4, 9, [1.0, 5.0, 9.0, 13.0]),
        ],
    )
    def test_fallback_parameters(
        self, every: int, max_labels: int, want: list[float]
    ) -> None:
        """Thirteen bands from 1, no zero: strides from 2 to 6 give 3 to 7 labels.

        The one nearest ``every`` among those within ``max_labels`` is taken.

        """
        levels = np.arange(1.0, 15.0)
        assert colorbar_ticks(levels, every=every, max_labels=max_labels) == want

    def test_no_stride_gives_three_labels(self) -> None:
        """Three bands, at most three labels: no stride fits, so the two ends."""
        assert colorbar_ticks([0.5, 1.5, 2.5, 3.5], max_labels=3) == [0.5, 3.5]

    def test_every_past_the_band_count(self) -> None:
        assert colorbar_ticks(np.arange(-3.0, 4.0), every=10) == [0.0]

    def test_end_levels_off_a_round_value_are_rounded(self) -> None:
        """Ends a hair past 0.4, as a snapped bar may have, label as 0.4."""
        levels = np.arange(-4, 5) * 0.1
        levels[0], levels[-1] = -0.4000000000000001, 0.4000000000000001
        assert colorbar_ticks(levels) == [-0.4, -0.2, 0.0, 0.2, 0.4]

    def test_bad_max_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="max_labels must be"):
            colorbar_ticks(np.arange(5.0), max_labels=2)


def test_truncate_cmap_endpoints() -> None:
    """Truncating maps the new endpoints onto the requested fractions."""
    cmap = plt.get_cmap("viridis")
    trunc = truncate_cmap(cmap, 0.25, 0.75)
    np.testing.assert_allclose(trunc(0.0)[:3], cmap(0.25)[:3], atol=0.02)
    np.testing.assert_allclose(trunc(1.0)[:3], cmap(0.75)[:3], atol=0.02)
