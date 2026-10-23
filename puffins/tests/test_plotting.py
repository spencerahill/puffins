"""Tests for the plotting module."""

from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from puffins.plotting import (  # noqa: E402
    band_colors_about_center,
    label_curve,
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


class TestLabelCurve:
    """Direct labels beside their curves, clear of every other.

    Every check is made on the rendered figure, in display pixels: the label's
    near edge lies ``gap`` points beyond its curve's extreme across the
    label's width and ``gap`` either side, every other curve lies on the far
    side of that extreme, the label lies inside the axes, and, measured from
    the label's box to each drawn line, it is nearer its own curve than any
    other.  Curve a rises from 0 to 3 across x from 0 to 10 and curve b falls
    from 3 to 0, so a lies above b only for x past 5, and by more the further
    right.

    """

    X = np.linspace(0.0, 10.0, 101)
    A = 0.3 * X
    B = 3.0 - 0.3 * X

    @pytest.fixture
    def axes(self) -> Iterator[matplotlib.axes.Axes]:
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        ax_.plot(self.X, self.A)
        ax_.plot(self.X, self.B)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        yield ax_
        plt.close(fig)

    @staticmethod
    def extent(ax_: matplotlib.axes.Axes, label: Text) -> matplotlib.transforms.Bbox:
        ax_.figure.canvas.draw()
        return label.get_window_extent()

    @staticmethod
    def segments(ax_: matplotlib.axes.Axes, x, y) -> list[np.ndarray]:
        """A curve's drawn segments in display pixels: one array of two points
        for each pair of neighbors that are both finite."""
        pts = ax_.transData.transform(np.column_stack([x, y]))
        ok = np.all(np.isfinite(pts), axis=1)
        return [pts[i : i + 2] for i in range(len(pts) - 1) if ok[i] and ok[i + 1]]

    def under(
        self, ax_: matplotlib.axes.Axes, box: matplotlib.transforms.Bbox, x, y, pad
    ) -> np.ndarray:
        """Display heights of a curve's drawn segments over the box's width
        widened by ``pad`` pixels either side: the parts of each segment lying
        within that range, by interpolation at the range's ends."""
        lo, hi = box.x0 - pad, box.x1 + pad
        heights = []
        for (x0, y0), (x1, y1) in self.segments(ax_, x, y):
            a, b = sorted((x0, x1))
            if b < lo or a > hi:
                continue
            for xv in (max(a, lo), min(b, hi)):
                heights.append(
                    y0 if x1 == x0 else y0 + (y1 - y0) * (xv - x0) / (x1 - x0)
                )
        return np.array(heights)

    def distance(self, ax_: matplotlib.axes.Axes, box, x, y) -> float:
        """Least distance, in pixels, from the box to a curve's drawn line,
        sampled every tenth of a pixel or finer along each segment."""
        best = np.inf
        for p, q in self.segments(ax_, x, y):
            n = max(int(np.hypot(*(q - p)) * 10), 2)
            s = p + np.linspace(0.0, 1.0, n)[:, None] * (q - p)
            dx = np.maximum(np.maximum(box.x0 - s[:, 0], s[:, 0] - box.x1), 0.0)
            dy = np.maximum(np.maximum(box.y0 - s[:, 1], s[:, 1] - box.y1), 0.0)
            best = min(best, float(np.hypot(dx, dy).min()))
        return best

    def check_beside(
        self, ax_, label, own, other, gap_pts: float = 2.0, above: bool = True, x=None
    ) -> matplotlib.transforms.Bbox:
        x = self.X if x is None else x
        box = self.extent(ax_, label)
        gap_px = gap_pts * ax_.figure.dpi / 72.0
        own_px = self.under(ax_, box, x, own, gap_px)
        if above:
            np.testing.assert_allclose(box.y0, own_px.max() + gap_px, atol=1e-6)
        else:
            np.testing.assert_allclose(box.y1, own_px.min() - gap_px, atol=1e-6)
        assert ax_.bbox.contains(box.x0, box.y0) and ax_.bbox.contains(box.x1, box.y1)
        own_d, other_d = (
            self.distance(ax_, box, x, own),
            self.distance(ax_, box, x, other),
        )
        assert gap_px - 1e-6 <= own_d < other_d
        return box

    def test_above_where_its_curve_leads_most(self, axes) -> None:
        """a leads b by more the further right, so the label ends at the frame.

        Its right edge lies within one data point, 0.1 in x, of the axes' right
        edge.

        """
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert label is not None
        box = self.check_beside(axes, label, self.A, self.B)
        point_px = axes.transData.transform([[0.0, 0.0], [0.1, 0.0]])
        assert box.x1 > axes.bbox.x1 - np.diff(point_px[:, 0])[0] - 1.0

    def test_where_the_gap_is_largest_not_the_curve_highest(self, axes) -> None:
        """Both curves rise, the labeled one by less, so it leads most at the left.

        The labeled curve, 1 + 0.3x, leads 0.35x by 1 - 0.05x, largest at x = 0,
        while the labeled curve itself is highest at x = 10.

        """
        own, other = 1.0 + 0.3 * self.X, 0.35 * self.X
        label = label_curve(axes, self.X, [own, other], 0, "rising", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_below(self, axes) -> None:
        label = label_curve(axes, self.X, [self.A, self.B], 1, "falling", side="below")
        assert label is not None
        self.check_beside(axes, label, self.B, self.A, above=False)

    def test_gap(self, axes) -> None:
        label = label_curve(
            axes, self.X, [self.A, self.B], 0, "rising", gap=10.0, side="above"
        )
        assert label is not None
        self.check_beside(axes, label, self.A, self.B, gap_pts=10.0)

    def test_stays_inside_the_axes(self, axes) -> None:
        """With the top at 3.1, the label must go where a is low enough."""
        axes.set_ylim(-1.0, 3.1)
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert label is not None
        self.check_beside(axes, label, self.A, self.B)

    def test_no_room_gives_none(self, axes) -> None:
        """A curve 0.3 below another everywhere, 11.6 pixels on this figure,
        leaves no room above it for a label 13.9 pixels tall plus ``gap``
        either side."""
        assert (
            label_curve(axes, self.X, [self.A, self.A + 0.3], 0, "low", side="above")
            is None
        )
        assert len(axes.texts) == 0

    def test_avoid(self, axes) -> None:
        first = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert first is not None
        second = label_curve(
            axes, self.X, [self.A, self.B], 0, "again", avoid=[first], side="above"
        )
        assert second is not None
        assert not self.extent(axes, second).overlaps(self.extent(axes, first))
        self.check_beside(axes, second, self.A, self.B)

    def test_text_kwargs(self, axes) -> None:
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", color="red")
        assert label is not None and label.get_color() == "red"

    def test_missing_points_of_the_labeled_curve(self, axes) -> None:
        """NaN in the labeled curve leaves no candidate there.

        A one-letter label would fit at the right edge, where a leads most,
        but a is missing past x = 8.

        """
        own = np.where(self.X > 8.0, np.nan, self.A)
        label = label_curve(axes, self.X, [own, self.B], 0, "a", side="above")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        assert x_left <= 8.0 + 1e-9

    def test_missing_points_of_another_curve(self, axes) -> None:
        """NaN in another curve is skipped, so it does not block a candidate.

        With b missing past x = 7, every candidate whose span, widened by
        ``gap`` times the square root of 2 (about 0.13 in x), lies past x = 7
        has no other curve beside it, and the leftmost of them, at x = 7.2, is
        taken.

        """
        other = np.where(self.X > 7.0, np.nan, self.B)
        label = label_curve(axes, self.X, [self.A, other], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(axes, label, self.A, other)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 7.2, atol=1e-6)

    def test_other_curve_counted_where_the_labeled_one_is_missing(self, axes) -> None:
        """Another curve's spike inside a gap in the labeled curve still blocks it.

        The labeled curve peaks at x = 5.1 and is missing at x = 5, where the
        other curve spikes to 3, above it; elsewhere the other curve is 0.  A
        one-letter label about 0.3 wide starting at x = 4.8 would reach the peak
        and cover the spike, and would win as the leftmost of the best
        positions were the spike overlooked.  Other curves are checked over
        the label's width plus about 0.13 in x either side (``gap`` times the
        square root of 2).  So a label starting at x = 5.1 would have the spike
        in that range, and one starting at 5.2 the spike's falling flank, at
        0.8, which cuts the lead to 1.6.  From 5.3 the flank is out of range,
        and the lead there, 2.39, is the largest.

        """
        own = np.where(np.isclose(self.X, 5.0), np.nan, 2.4 - 0.1 * abs(self.X - 5.1))
        other = np.where(np.isclose(self.X, 5.0), 3.0, 0.0)
        label = label_curve(axes, self.X, [own, other], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other)
        (x_left, _), (x_right, _) = axes.transData.inverted().transform(
            [[box.x0, box.y0], [box.x1, box.y1]]
        )
        assert 0.2 < x_right - x_left < 0.9  # the premise above
        np.testing.assert_allclose(x_left, 5.3, atol=1e-6)

    def test_tie_goes_to_the_leftmost(self, axes) -> None:
        """Two flat curves lead by the same everywhere, so the label starts at 0."""
        own, other = np.full_like(self.X, 2.0), np.full_like(self.X, 1.0)
        label = label_curve(axes, self.X, [own, other], 0, "flat")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_log_scale(self, axes) -> None:
        """Distances are in display space, so a log axis is handled too."""
        axes.set_yscale("log")
        axes.set_ylim(0.1, 20.0)
        own, other = self.A + 0.2, self.B + 0.2
        label = label_curve(axes, self.X, [own, other], 0, "rising", side="above")
        assert label is not None
        self.check_beside(axes, label, own, other)

    def test_inverted_x_axis(self, axes) -> None:
        axes.invert_xaxis()
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert label is not None
        self.check_beside(axes, label, self.A, self.B)

    def test_segment_entering_between_points(self) -> None:
        """An other curve rising between two points still blocks the label.

        At x = 0 the other curve is 0 and at x = 4 it is 3, above the labeled
        curve's 2.  The label, about three units wide, starting at x = 0 would
        overlap that segment, which only the values interpolated at its right
        edge reveal; so it starts at x = 8.

        """
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.array([0.0, 4.0, 8.0])
        own, other = np.array([2.0, 2.0, 2.0]), np.array([0.0, 3.0, 0.0])
        ax_.set_xlim(0.0, 12.0)
        ax_.set_ylim(-1.0, 5.0)
        label = label_curve(ax_, x, [own, other], 0, "a long label", fontsize=12)
        assert label is not None
        box = self.extent(ax_, label)
        (x_left, _), (x_right, _) = ax_.transData.inverted().transform(
            [[box.x0, box.y0], [box.x1, box.y1]]
        )
        assert 2.7 < x_right - x_left < 3.9  # the premise above
        np.testing.assert_allclose(x_left, 8.0, atol=1e-6)
        plt.close(fig)

    @staticmethod
    def flat_axes(levels: list[float], ylim: tuple[float, float]):
        """A figure of flat curves at ``levels``, x from 0 to 10."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 101)
        curves = [np.full_like(x, v) for v in levels]
        for c in curves:
            ax_.plot(x, c)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(*ylim)
        return fig, ax_, x, curves

    def test_steep_curve_just_past_the_label(self) -> None:
        """A curve rising steeply just beyond the label's edge keeps clear of it.

        The labeled curve is flat at 1; the other falls slowly to 0 at x = 5,
        then rises at 300 per unit x to 3.  Without the margin beside the
        label's width, the first version of this function started the label
        at x = 4.25, ending just short of x = 5, and 0.36 pixels from that
        rise.

        """
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 1001)
        own = np.ones_like(x)
        other = np.where(x <= 5.0, 0.05 * (5.0 - x), np.minimum(300.0 * (x - 5.0), 3.0))
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 4.0)
        label = label_curve(ax_, x, [own, other], 0, "flat", side="above")
        assert label is not None
        self.check_beside(ax_, label, own, other, x=x)
        plt.close(fig)

    def test_own_flank_just_past_the_label(self) -> None:
        """The labeled curve's own steep flank beside the label keeps ``gap`` too."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 101)
        own = 4.0 / (1.0 + np.exp(-4.0 * (x - 5.0)))
        other = np.full_like(x, -0.5)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 4.1)
        label = label_curve(ax_, x, [own, other], 0, "rising", side="above")
        assert label is not None
        box = self.extent(ax_, label)
        assert self.distance(ax_, box, x, own) >= 2.0 * ax_.figure.dpi / 72.0 - 1.0
        plt.close(fig)

    def test_gap_in_the_labeled_curve_is_not_bridged(self) -> None:
        """No curve is counted across a NaN gap, so no label hangs over one.

        The labeled curve is 1 up to x = 5 and 4 from x = 9.95, missing
        between.  The other is 1.35 everywhere: 13.5 pixels above the labeled
        curve up to x = 5, too little for a label, and too far right past
        x = 9.95 for one to fit.  A bridge from 1 to 4 across the gap would
        rise past 1.35 with room to spare.

        """
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 201)
        own = np.where(x <= 5.0, 1.0, np.where(x >= 9.95, 4.0, np.nan))
        other = np.full_like(x, 1.35)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        assert label_curve(ax_, x, [own, other], 0, "a", side="above") is None
        plt.close(fig)

    def test_other_curve_not_extended_past_its_end(self) -> None:
        """A curve missing past x = 2 does not block labels beyond x = 2."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 101)
        own = 0.3 * x
        other = np.where(x < 2.0, 4.5, np.nan)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        label = label_curve(ax_, x, [own, other], 0, "rising", side="above")
        assert label is not None
        self.check_beside(ax_, label, own, other, x=x)
        plt.close(fig)

    def test_middle_curve(self) -> None:
        """Flat curves at 0, 1 and 2: the middle one is labeled between two others.

        Above and below it, the nearest other curve is equally far, so the tie
        goes to above.

        """
        fig, ax_, x, curves = self.flat_axes([0.0, 1.0, 2.0], (-1.0, 3.0))
        label = label_curve(ax_, x, curves, 1, "middle")
        assert label is not None
        box = self.extent(ax_, label)
        d = [self.distance(ax_, box, x, c) for c in curves]
        assert d[1] < min(d[0], d[2]) and d[1] >= 2.0 * 100.0 / 72.0 - 1e-6
        assert box.y0 > ax_.transData.transform((0.0, 1.0))[1]
        plt.close(fig)

    def test_toward_the_neighbor_only_when_away_does_not_fit(self) -> None:
        """Flat curves at 0 and 4, axes bottom at -0.1: below the lower curve
        there is no room, so its label goes above it, still nearer to it."""
        fig, ax_, x, curves = self.flat_axes([0.0, 4.0], (-0.1, 5.0))
        label = label_curve(ax_, x, curves, 0, "lower")
        assert label is not None
        box = self.extent(ax_, label)
        assert box.y0 > ax_.transData.transform((0.0, 0.0))[1]
        assert self.distance(ax_, box, x, curves[0]) < self.distance(
            ax_, box, x, curves[1]
        )
        plt.close(fig)

    def test_auto_takes_the_side_away_from_the_nearest_curve(self, axes) -> None:
        """a = 0.3x and b = 4 - 0.25x are farthest apart at x = 0, with a below,
        so a's label goes below it at the left."""
        b = 4.0 - 0.25 * self.X
        label = label_curve(axes, self.X, [self.A, b], 0, "rising")
        assert label is not None
        box = self.check_beside(axes, label, self.A, b, above=False)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_masked_log_axis(self, axes) -> None:
        """On a log axis that masks nonpositive values, the label is still placed."""
        axes.set_yscale("log", nonpositive="mask")
        axes.set_ylim(0.1, 20.0)
        own, other = self.A + 0.2, self.B + 0.2
        label = label_curve(axes, self.X, [own, other], 0, "rising", side="above")
        assert label is not None
        self.check_beside(axes, label, own, other)

    def test_dates(self) -> None:
        """Times as datetime64, as xarray decodes them, are placed on the date axis."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        t = np.arange("2000-01", "2010-01", dtype="datetime64[M]").astype(
            "datetime64[ns]"
        )
        own, other = np.linspace(0.0, 3.0, t.size), np.linspace(3.0, 0.0, t.size)
        ax_.plot(t, own)
        ax_.plot(t, other)
        ax_.set_ylim(-1.0, 5.0)
        label = label_curve(ax_, t, [own, other], 0, "rising", side="above")
        assert label is not None
        box = self.extent(ax_, label)
        x_days = ax_.xaxis.convert_units(t)
        assert self.distance(ax_, box, x_days, own) < self.distance(
            ax_, box, x_days, other
        )
        plt.close(fig)

    def test_none_in_avoid_is_skipped(self, axes) -> None:
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", avoid=[None])
        assert label is not None

    def test_error_leaves_nothing_on_the_axes(self, axes) -> None:
        not_a_label: Any = "not a label"
        with pytest.raises(AttributeError):
            label_curve(
                axes, self.X, [self.A, self.B], 0, "rising", avoid=[not_a_label]
            )
        assert len(axes.texts) == 0

    @pytest.mark.parametrize("key", ["ha", "va", "transform", "rotation_mode", "bbox"])
    def test_keywords_set_here_raise(self, axes, key: str) -> None:
        kwargs: dict[str, Any] = {key: None}
        with pytest.raises(ValueError, match="sets or forbids"):
            label_curve(axes, self.X, [self.A, self.B], 0, "t", **kwargs)
        assert len(axes.texts) == 0

    def test_single_curve_goes_leftmost(self, axes) -> None:
        """With no other curve, every candidate ties, so the label starts at 0."""
        label = label_curve(axes, self.X, [self.A], 0, "alone", side="above")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_all_missing_other_curve_is_ignored(self, axes) -> None:
        ys = [self.A, np.full_like(self.X, np.nan)]
        label = label_curve(axes, self.X, ys, 0, "rising", side="above")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_identical_curves_give_none(self, axes) -> None:
        """A label for one of two equal curves is as near the other."""
        assert label_curve(axes, self.X, [self.A, self.A.copy()], 0, "twin") is None
        assert len(axes.texts) == 0

    def test_small_lead_on_parallel_slopes_goes_leftmost(self, axes) -> None:
        """0.3x + 0.05 leads 0.3x by 1.9 pixels, and by 1.5 over the spans
        compared, the other curve's being the wider (1.9 less 0.37 pixels a
        pixel of slope times the 1.15 pixels the spans differ by).  That is
        above the 1.15 needed, ``gap`` times the square root of 2 less
        ``gap``.

        The curves run on to x = 12, past the axes, so no span is cut short at
        an end of the data, and the leads, equal but for rounding, tie: the
        label starts at 0.

        """
        x = np.linspace(0.0, 12.0, 121)
        own, other = 0.3 * x + 0.05, 0.3 * x
        label = label_curve(axes, x, [own, other], 0, "upper", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other, x=x)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_sparse_gap_in_another_curve(self, axes) -> None:
        """The other curve is missing at x = 5 only, with points a unit apart.

        A label starting at 5 has, over its span, no part of the other curve
        drawn, so it has no curve beside it and is preferred to any where the
        other curve is 2 below.  The segments either side of the missing point
        are not drawn, and must not be interpolated, which gives NaN.

        """
        x = np.arange(11.0)
        own = np.full_like(x, 2.0)
        other = np.where(x == 5.0, np.nan, 0.0)
        label = label_curve(axes, x, [own, other], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other, x=x)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 5.0, atol=1e-6)

    def test_away_side_judged_by_the_nearest_curve(self, axes) -> None:
        """The labeled curve is flat at 2, between a curve below and one above.

        Left of x = 5 they are 50 and 51.5 pixels from it, so the nearest is
        below and above is the side away from it.  Right of 5 they are 40 and
        100.  Above, the nearest curve is 50 pixels away on the left and 40 on
        the right, so the label starts at 0.

        """
        per_px = 6.0 / axes.bbox.height
        left = self.X < 5.0
        own = np.full_like(self.X, 2.0)
        below = 2.0 - per_px * np.where(left, 50.0, 40.0)
        above = 2.0 + per_px * np.where(left, 51.5, 100.0)
        label = label_curve(axes, self.X, [own, below, above], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, below)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_nearest_of_two_curves_ahead_decides(self, axes) -> None:
        """Two curves above the flat labeled curve, none below.

        Left of x = 5 they are 30 and 60 pixels above it, and right of 5, 45
        and 50.  The nearer is 30 on the left and 45 on the right, so the label
        starts at the first point whose span, widened by about 0.13 in x, lies
        past x = 5, which is 5.2.

        """
        per_px = 6.0 / axes.bbox.height
        left = self.X < 5.0
        own = np.full_like(self.X, 2.0)
        nearer = 2.0 + per_px * np.where(left, 30.0, 45.0)
        farther = 2.0 + per_px * np.where(left, 60.0, 50.0)
        label = label_curve(axes, self.X, [own, nearer, farther], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, nearer)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 5.2, atol=1e-6)

    @pytest.mark.parametrize(
        ("other_name", "placed"),
        [
            ("behind by 25 pixels", True),
            ("behind by 17.5 pixels", False),
            ("ahead by 17.5 pixels", False),
            ("ahead by 10 pixels, slanting away", False),
        ],
    )
    def test_other_curve_against_the_corner_distance(
        self, other_name: str, placed: bool
    ) -> None:
        """Another curve nearer the label than its own curve's corner blocks it.

        Data units here are display pixels: axes 200 by 300 pixels, x shown
        from 101 to 301.  The labeled curve falls from 150 at x = 100 to 100 at
        x = 120 and is flat to x = 300.  With a gap of 10 points, 13.9 pixels,
        a label above it can start only at x = 120: at 100 it would start
        outside the axes, and at 300 it would end outside them.  The labeled
        curve's extreme over the label's span is on the falling stretch, at
        x = 120 - gap, so the label's near edge is at 100 + 3.5 gap, and the
        curve comes within 18.05 pixels of the label's lower left corner, at
        x = 103.2.  The other curve is flat, 25 or 17.5 pixels below the
        label's near edge or 17.5 above its far edge, or falls from 300 at
        x = 100 to 10 pixels above the far edge at x = 120.  Only at 25 pixels
        is it farther from the label than the labeled curve.

        """
        fig = plt.figure(figsize=(4.0, 3.0), dpi=100)
        ax_ = fig.add_axes((0.0, 0.0, 0.5, 1.0))
        ax_.set_xlim(101.0, 301.0)
        ax_.set_ylim(0.0, 300.0)
        x = np.array([100.0, 120.0, 300.0])
        own = np.array([150.0, 100.0, 100.0])
        gap_px = 10.0 * 100.0 / 72.0
        near_edge = 100.0 + 3.5 * gap_px
        probe = ax_.text(0.0, 0.0, "a")
        fig.canvas.draw()
        far_edge = near_edge + probe.get_window_extent().height
        probe.remove()
        other = {
            "behind by 25 pixels": np.full(3, near_edge - 25.0),
            "behind by 17.5 pixels": np.full(3, near_edge - 17.5),
            "ahead by 17.5 pixels": np.full(3, far_edge + 17.5),
            "ahead by 10 pixels, slanting away": np.array(
                [300.0, far_edge + 10.0, far_edge + 10.0]
            ),
        }[other_name]
        label = label_curve(ax_, x, [own, other], 0, "a", side="above", gap=10.0)
        if placed:
            assert label is not None
            box = self.check_beside(ax_, label, own, other, gap_pts=10.0, x=x)
            np.testing.assert_allclose(
                self.distance(ax_, box, x, own), 18.05, atol=0.01
            )
        else:
            assert label is None
        plt.close(fig)

    def test_nearest_of_two_curves_behind_decides(self, axes) -> None:
        """The labeled curve is 2; behind it are 1 and 1.5 - 0.1x.

        The nearer of the two is 1.5 - 0.1x up to x = 5 and 1 after, so the
        lead rises to 1 and stays there: the label starts at the first point
        whose span, widened by about 0.13 in x, lies past x = 5, which is 5.2.

        """
        own = np.full_like(self.X, 2.0)
        ys = [own, np.full_like(self.X, 1.0), 1.5 - 0.1 * self.X]
        label = label_curve(axes, self.X, ys, 0, "a", side="above")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 5.2, atol=1e-6)

    def test_slowly_rising_lead_goes_right(self, axes) -> None:
        """A lead rising by 0.02 pixels a point is not taken for a tie, so the
        label ends within one point, 0.1 in x, of the axes' right edge."""
        own = 2.0 + 0.005 * self.X
        label = label_curve(
            axes, self.X, [own, np.zeros_like(self.X)], 0, "a", side="above"
        )
        assert label is not None
        box = self.extent(axes, label)
        point_px = np.diff(axes.transData.transform([[0.0, 0.0], [0.1, 0.0]])[:, 0])[0]
        assert box.x1 > axes.bbox.x1 - point_px - 1e-6

    def test_nearest_of_several_curves_decides(self, axes) -> None:
        """With a third curve 0.3 above a, 11.6 pixels, no label fits above a,
        though b alone would allow one."""
        ys = [self.A, self.B, self.A + 0.3]
        assert label_curve(axes, self.X, ys, 0, "rising", side="above") is None

    def test_vertical_segment_counted(self) -> None:
        """A drop drawn at a repeated x blocks a label over it.

        The other curve falls from 3 to 0 at x = 5, a vertical segment, and is
        missing before; the labeled curve peaks at 2.9 there.  With x shown
        from 4, the only candidate inside the axes starts at x = 5, where only
        the top of the drop, at a repeated point, reaches above the peak.

        """
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.array([0.0, 5.0, 5.0, 10.0])
        own, other = np.array([1.0, 2.9, 2.9, 1.0]), np.array([np.nan, 3.0, 0.0, 0.0])
        ax_.set_xlim(4.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        assert label_curve(ax_, x, [own, other], 0, "a", side="above") is None
        plt.close(fig)

    def test_candidates_only_where_the_labeled_curve_is(self, axes) -> None:
        """b, labeled below, is missing past x = 8, so its label starts at 8.

        With a gap of 10 points, a label starting at 8.4 would still have
        b's last points within its span, and would win there, a leading b by
        more the further right.

        """
        own = np.where(self.X > 8.0, np.nan, self.B)
        label = label_curve(axes, self.X, [self.A, own], 1, "f", side="below", gap=10.0)
        assert label is not None
        box = self.check_beside(axes, label, own, self.A, gap_pts=10.0, above=False)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 8.0, atol=1e-6)

    @pytest.mark.parametrize("far_first", [True, False])
    def test_avoid_several(self, axes, far_first: bool) -> None:
        """Every text in ``avoid`` is checked, whatever its order."""
        first = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert first is not None
        far = axes.text(0.5, 4.5, "far")
        avoid = [far, first] if far_first else [first, far]
        second = label_curve(
            axes, self.X, [self.A, self.B], 0, "again", avoid=avoid, side="above"
        )
        assert second is not None
        assert not self.extent(axes, second).overlaps(self.extent(axes, first))

    def test_fontsize_sets_the_measured_size(self, axes) -> None:
        """At 24 points the label is wider, and must still end inside the axes."""
        label = label_curve(
            axes, self.X, [self.A, self.B], 0, "rising", side="above", fontsize=24
        )
        assert label is not None and label.get_fontsize() == 24
        self.check_beside(axes, label, self.A, self.B)

    def test_dpi(self) -> None:
        """The gap is in points, so it scales with the figure's dpi."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=200)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        label = label_curve(
            ax_, self.X, [self.A, self.B], 0, "r", side="above", gap=10.0
        )
        assert label is not None
        self.check_beside(ax_, label, self.A, self.B, gap_pts=10.0)
        plt.close(fig)

    def test_layout_settled_by_a_draw(self) -> None:
        """An equal aspect resizes the axes only when drawn, before measuring."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        ax_.set_aspect("equal")
        label = label_curve(
            ax_, self.X, [self.A, self.B], 0, "rising", side="above", gap=10.0
        )
        assert label is not None
        self.check_beside(ax_, label, self.A, self.B, gap_pts=10.0)
        plt.close(fig)

    def test_left_edge_of_the_axes(self, axes) -> None:
        """With x shown from 2, the lead 1 - 0.05x is largest at x = 2."""
        axes.set_xlim(2.0, 10.0)
        own, other = 1.0 + 0.3 * self.X, 0.35 * self.X
        label = label_curve(axes, self.X, [own, other], 0, "rising", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 2.0, atol=1e-6)

    def test_bottom_edge_of_the_axes(self, axes) -> None:
        """With y shown from -0.3, a label below b must stay above that."""
        axes.set_ylim(-0.3, 5.0)
        label = label_curve(axes, self.X, [self.A, self.B], 1, "falling", side="below")
        assert label is not None
        self.check_beside(axes, label, self.B, self.A, above=False)

    def test_logit_axis_with_integer_x(self) -> None:
        """Integer x and a logit axis, which has no y at 0."""
        fig, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.arange(11)
        own, other = 0.1 + 0.08 * x, np.full(x.shape, 0.03)
        ax_.set_yscale("logit")
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(0.01, 0.99)
        label = label_curve(ax_, x, [own, other], 0, "rising", side="above")
        assert label is not None
        self.check_beside(ax_, label, own, other, x=x)
        plt.close(fig)

    @pytest.mark.parametrize("gap", [0.0, -1.0])
    def test_nonpositive_gap_raises(self, axes, gap: float) -> None:
        with pytest.raises(ValueError, match="gap must be positive"):
            label_curve(axes, self.X, [self.A, self.B], 0, "t", gap=gap)

    @pytest.mark.parametrize(
        ("x", "ys", "which", "side", "match"),
        [
            (np.arange(3.0), [np.zeros(3), np.ones(3)], 0, "left", "side must be"),
            (np.arange(3.0), [np.zeros(3), np.ones(3)], 2, "above", "which must"),
            (np.arange(3.0), [np.zeros(3), np.ones(3)], -1, "above", "which must"),
            (np.zeros((3, 2)), [np.zeros(3)], 0, "above", "curve must"),
            (np.zeros((3, 2)), [np.zeros((3, 2))], 0, "above", "one-dimensional"),
            (np.arange(3.0), [np.zeros(4), np.zeros(3)], 0, "above", "curve must"),
        ],
    )
    def test_bad_arguments_raise(
        self,
        axes: matplotlib.axes.Axes,
        x: np.ndarray,
        ys: list[np.ndarray],
        which: int,
        side: str,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            label_curve(axes, x, ys, which, "t", side=side)


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


def test_truncate_cmap_endpoints() -> None:
    """Truncating maps the new endpoints onto the requested fractions."""
    cmap = plt.get_cmap("viridis")
    trunc = truncate_cmap(cmap, 0.25, 0.75)
    np.testing.assert_allclose(trunc(0.0)[:3], cmap(0.25)[:3], atol=0.02)
    np.testing.assert_allclose(trunc(1.0)[:3], cmap(0.75)[:3], atol=0.02)
