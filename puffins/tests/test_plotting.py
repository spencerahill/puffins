"""Tests for the plotting module."""

import copy
import io
from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from puffins.plotting import (  # noqa: E402
    band_colors_about_center,
    colorbar_ticks,
    label_curve,
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


class TestLabelCurve:
    """Direct labels beside their curves, clear of every other.

    Every check is made on the rendered figure, in display pixels, and on the
    label's box as a saved PDF lays it out: the label's near edge lies
    ``gap`` points beyond its curve's extreme across the label's width and
    ``gap`` either side, every other curve lies on the far side of that
    extreme, the label lies inside the axes, and, measured from the label's
    box to each drawn line, it is nearer its own curve than any other.  Curve a rises from 0 to 3 across x from 0 to 10 and curve b falls
    from 3 to 0, so a lies above b only for x past 5, and by more the further
    right.

    """

    X = np.linspace(0.0, 10.0, 101)
    A = 0.3 * X
    B = 3.0 - 0.3 * X

    @pytest.fixture(autouse=True)
    def close_figures(self) -> Iterator[None]:
        """Close every figure a test opens, also when the test fails."""
        yield
        plt.close("all")

    @pytest.fixture
    def axes(self) -> matplotlib.axes.Axes:
        _, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        ax_.plot(self.X, self.A)
        ax_.plot(self.X, self.B)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        return ax_

    @staticmethod
    def extent(ax_: matplotlib.axes.Axes, label: Text) -> matplotlib.transforms.Bbox:
        ax_.figure.canvas.draw()
        return label.get_window_extent()

    @staticmethod
    def pdf_extent(
        ax_: matplotlib.axes.Axes, label: Text
    ) -> matplotlib.transforms.Bbox:
        """The label's box as matplotlib's PDF backend lays it out, at 72 dpi,
        in the figure's display pixels; measured on a copy of the label."""
        fig = ax_.get_figure(root=True)
        assert fig is not None
        width, height = fig.get_size_inches() * 72.0
        renderer = RendererPdf(PdfFile(io.BytesIO()), 72, height, width)
        box = copy.copy(label).get_window_extent(renderer=renderer, dpi=72.0)
        return matplotlib.transforms.Bbox(box.get_points() * fig.dpi / 72.0)

    @staticmethod
    def segments(
        ax_: matplotlib.axes.Axes, x: np.ndarray, y: np.ndarray
    ) -> list[np.ndarray]:
        """A curve's drawn segments in display pixels: one array of two points
        for each pair of neighbors that are both finite."""
        pts = ax_.transData.transform(np.column_stack([x, y]))
        ok = np.all(np.isfinite(pts), axis=1)
        return [pts[i : i + 2] for i in range(len(pts) - 1) if ok[i] and ok[i + 1]]

    @staticmethod
    def segment_box_distance(
        p: np.ndarray, q: np.ndarray, box: matplotlib.transforms.Bbox
    ) -> float:
        """Exact least distance, in pixels, from the segment pq to the box.

        Zero where the segment meets the box, found by clipping the segment
        to the box.  Otherwise the least distance between the two convex
        shapes lies at a vertex of one of them: an end of the segment, or a
        corner of the box.

        """
        t0, t1 = 0.0, 1.0
        for start, step, lo, hi in (
            (p[0], q[0] - p[0], box.x0, box.x1),
            (p[1], q[1] - p[1], box.y0, box.y1),
        ):
            if step == 0.0:
                if not lo <= start <= hi:
                    t0, t1 = 1.0, 0.0
            else:
                ta, tb = (lo - start) / step, (hi - start) / step
                t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 <= t1:
            return 0.0
        dists = [
            float(
                np.hypot(
                    max(box.x0 - end[0], 0.0, end[0] - box.x1),
                    max(box.y0 - end[1], 0.0, end[1] - box.y1),
                )
            )
            for end in (p, q)
        ]
        d = q - p
        if np.dot(d, d) > 0.0:
            for corner in box.corners():
                t = np.clip(np.dot(corner - p, d) / np.dot(d, d), 0.0, 1.0)
                dists.append(float(np.hypot(*(p + t * d - corner))))
        return min(dists)

    def under(
        self,
        ax_: matplotlib.axes.Axes,
        box: matplotlib.transforms.Bbox,
        x: np.ndarray,
        y: np.ndarray,
        pad: float,
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

    def distance(
        self,
        ax_: matplotlib.axes.Axes,
        box: matplotlib.transforms.Bbox,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Exact least distance, in pixels, from the box to a curve's drawn line."""
        return min(
            (self.segment_box_distance(p, q, box) for p, q in self.segments(ax_, x, y)),
            default=np.inf,
        )

    def check_beside(
        self,
        ax_: matplotlib.axes.Axes,
        label: Text,
        own: np.ndarray,
        other: np.ndarray,
        gap_pts: float = 2.0,
        above: bool = True,
        x: np.ndarray | None = None,
    ) -> matplotlib.transforms.Bbox:
        x = self.X if x is None else x
        box = self.extent(ax_, label)
        in_pdf = self.pdf_extent(ax_, label)
        gap_px = gap_pts * ax_.figure.dpi / 72.0
        both = matplotlib.transforms.Bbox.union([box, in_pdf])
        own_px = self.under(ax_, both, x, own, gap_px)
        for drawn in (box, in_pdf):
            if above:
                np.testing.assert_allclose(drawn.y0, own_px.max() + gap_px, atol=1e-6)
            else:
                np.testing.assert_allclose(drawn.y1, own_px.min() - gap_px, atol=1e-6)
            assert ax_.bbox.contains(drawn.x0, drawn.y0)
            assert ax_.bbox.contains(drawn.x1, drawn.y1)
            own_d, other_d = (
                self.distance(ax_, drawn, x, own),
                self.distance(ax_, drawn, x, other),
            )
            assert gap_px - 1e-6 <= own_d < other_d
        return box

    def test_above_where_its_curve_leads_most(self, axes: matplotlib.axes.Axes) -> None:
        """a leads b by more the further right, so the label ends at the frame.

        Its right edge lies within one data point, 0.1 in x, of the axes' right
        edge.

        """
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert label is not None
        box = self.check_beside(axes, label, self.A, self.B)
        point_px = axes.transData.transform([[0.0, 0.0], [0.1, 0.0]])
        assert box.x1 > axes.bbox.x1 - np.diff(point_px[:, 0])[0] - 1.0

    def test_where_the_gap_is_largest_not_the_curve_highest(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    def test_below(self, axes: matplotlib.axes.Axes) -> None:
        label = label_curve(axes, self.X, [self.A, self.B], 1, "falling", side="below")
        assert label is not None
        self.check_beside(axes, label, self.B, self.A, above=False)

    def test_gap(self, axes: matplotlib.axes.Axes) -> None:
        label = label_curve(
            axes, self.X, [self.A, self.B], 0, "rising", gap=10.0, side="above"
        )
        assert label is not None
        self.check_beside(axes, label, self.A, self.B, gap_pts=10.0)

    def test_stays_inside_the_axes(self, axes: matplotlib.axes.Axes) -> None:
        """With the top at 3.1, the label must go where a is low enough."""
        axes.set_ylim(-1.0, 3.1)
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert label is not None
        self.check_beside(axes, label, self.A, self.B)

    def test_no_room_gives_none(self, axes: matplotlib.axes.Axes) -> None:
        """A curve 0.3 below another everywhere, 11.6 pixels on this figure,
        leaves no room above it for a label 13.9 pixels tall plus ``gap``
        either side."""
        assert (
            label_curve(axes, self.X, [self.A, self.A + 0.3], 0, "low", side="above")
            is None
        )
        assert len(axes.texts) == 0

    def test_avoid(self, axes: matplotlib.axes.Axes) -> None:
        first = label_curve(axes, self.X, [self.A, self.B], 0, "rising", side="above")
        assert first is not None
        second = label_curve(
            axes, self.X, [self.A, self.B], 0, "again", avoid=[first], side="above"
        )
        assert second is not None
        assert not self.extent(axes, second).overlaps(self.extent(axes, first))
        self.check_beside(axes, second, self.A, self.B)

    def test_text_kwargs(self, axes: matplotlib.axes.Axes) -> None:
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", color="red")
        assert label is not None and label.get_color() == "red"

    def test_missing_points_of_the_labeled_curve(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    def test_missing_points_of_another_curve(self, axes: matplotlib.axes.Axes) -> None:
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

    def test_other_curve_counted_where_the_labeled_one_is_missing(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    def test_tie_goes_to_the_leftmost(self, axes: matplotlib.axes.Axes) -> None:
        """Two flat curves lead by the same everywhere, so the label starts at 0."""
        own, other = np.full_like(self.X, 2.0), np.full_like(self.X, 1.0)
        label = label_curve(axes, self.X, [own, other], 0, "flat")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_log_scale(self, axes: matplotlib.axes.Axes) -> None:
        """Distances are in display space, so a log axis is handled too."""
        axes.set_yscale("log")
        axes.set_ylim(0.1, 20.0)
        own, other = self.A + 0.2, self.B + 0.2
        label = label_curve(axes, self.X, [own, other], 0, "rising", side="above")
        assert label is not None
        self.check_beside(axes, label, own, other)

    def test_inverted_x_axis(self, axes: matplotlib.axes.Axes) -> None:
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

    @staticmethod
    def flat_axes(
        levels: list[float], ylim: tuple[float, float]
    ) -> tuple[
        matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray, list[np.ndarray]
    ]:
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
        assert self.distance(ax_, box, x, own) >= 2.0 * ax_.figure.dpi / 72.0 - 1e-6

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

    @pytest.mark.parametrize("missing", ["nan y", "masked y", "nan x", "masked x"])
    def test_lone_point_is_no_curve(self, missing: str) -> None:
        """A point between two missing ones is drawn as nothing, so it counts
        as no curve.

        The labeled curve is flat at 1 but for a point of 3 at x = 5, whose
        neighbors are missing; the other curve is flat at 2.5.  Counted, the
        lone point would put the label above it, over the other curve and 118
        pixels from the drawn line at 1.  Not counted, the two curves are
        equally far apart everywhere, and the label starts at x = 0.

        """
        _, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 101)
        own = np.where(np.isclose(x, 5.0), 3.0, 1.0)
        other = np.full_like(x, 2.5)
        gone = np.isclose(x, 4.9) | np.isclose(x, 5.1)
        x_in: np.ndarray = x
        own_in: np.ndarray = own
        if missing == "nan y":
            own_in = np.where(gone, np.nan, own)
        elif missing == "masked y":
            own_in = np.ma.masked_where(gone, own)
        elif missing == "nan x":
            x_in = np.where(gone, np.nan, x)
        else:
            x_in = np.ma.masked_where(gone, x)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(0.0, 4.0)
        label = label_curve(ax_, x_in, [own_in, other], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(ax_, label, own, other, x=np.where(gone, np.nan, x))
        x_left = ax_.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_masked_points_of_another_curve(self) -> None:
        """Masked points are drawn as gaps, so they block nothing.

        The other curve is 1.2, too near the labeled curve at 1 for a label
        between them, and is masked past x = 5.  The label starts at the first
        point whose span, widened by about 0.13 in x, lies past x = 5, which
        is 5.2.

        """
        _, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.linspace(0.0, 10.0, 101)
        own = np.ones_like(x)
        other = np.ma.masked_where(x > 5.0, np.full_like(x, 1.2))
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(0.0, 4.0)
        label = label_curve(ax_, x, [own, other], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(ax_, label, own, np.ma.filled(other, np.nan), x=x)
        x_left = ax_.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 5.2, atol=1e-6)

    def test_decreasing_x_with_a_repeated_value(self) -> None:
        """A step given right to left is placed as the same step left to right.

        x runs 10, 5, 5, 0, so the labeled curve is 1 right of x = 5 and 3 left
        of it, joined by a drop at x = 5; the other curve is 0.  Sorting the
        points by x would join (0, 3) to (5, 1) and (5, 3) to (10, 1), two
        segments never drawn, and put the label 60 pixels from its curve.

        """
        _, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        x = np.array([10.0, 5.0, 5.0, 0.0])
        own, other = np.array([1.0, 1.0, 3.0, 3.0]), np.zeros(4)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-0.5, 4.0)
        boxes = []
        for order in (slice(None), slice(None, None, -1)):
            label = label_curve(
                ax_,
                x[order],
                [own[order], other[order]],
                0,
                "a long label",
                side="below",
            )
            assert label is not None
            boxes.append(self.check_beside(ax_, label, own, other, above=False, x=x))
            label.remove()
        np.testing.assert_allclose(
            boxes[0].get_points(), boxes[1].get_points(), atol=1e-6
        )

    @pytest.mark.parametrize(
        ("levels", "ylim"),
        [
            ([0.0, 1.0, 2.0], (-1.0, 3.0)),
            ([0.15, 2 * 0.15, 3 * 0.15], (0.0, 0.6)),
        ],
    )
    def test_middle_curve(self, levels: list[float], ylim: tuple[float, float]) -> None:
        """Three equally spaced flat curves: the middle one is labeled between
        two others.

        Above and below it, the nearest other curve is equally far, so the tie
        goes to above.  At 0.15, 0.3 and 0.45 the gaps on the screen differ by
        rounding, 57.75 pixels below and 57.74999999999997 above, which is
        still a tie.

        """
        fig, ax_, x, curves = self.flat_axes(levels, ylim)
        label = label_curve(ax_, x, curves, 1, "middle")
        assert label is not None
        box = self.extent(ax_, label)
        d = [self.distance(ax_, box, x, c) for c in curves]
        assert d[1] < min(d[0], d[2]) and d[1] >= 2.0 * 100.0 / 72.0 - 1e-6
        assert box.y0 > ax_.transData.transform((0.0, levels[1]))[1]

    def test_given_side_ranked_by_clearance_alone(self) -> None:
        """With the side given, the x is where the nearest other curve is
        farthest, whichever side of the labeled curve that curve is on.

        The labeled curve is flat at 2.  The other curve is 8 pixels below it
        left of x = 5 and 120 pixels above it from x = 5.  Above the labeled
        curve, the label fits under the far curve, starting at the first
        point whose span, widened by about 0.13 in x, lies past the rise at
        x = 5, which is 5.2.  Ranked first by the side away from the nearest
        curve, it went to x = 0, 10.8 pixels from the other curve.

        """
        _, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=100)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        per_px = 6.0 / ax_.bbox.height
        own = np.full_like(self.X, 2.0)
        other = np.where(self.X < 5.0, 2.0 - per_px * 8.0, 2.0 + per_px * 120.0)
        label = label_curve(ax_, self.X, [own, other], 0, "a", side="above")
        assert label is not None
        box = self.check_beside(ax_, label, own, other)
        x_left = ax_.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 5.2, atol=1e-6)

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

    def test_auto_takes_the_side_away_from_the_nearest_curve(
        self, axes: matplotlib.axes.Axes
    ) -> None:
        """a = 0.3x and b = 4 - 0.25x are farthest apart at x = 0, with a below,
        so a's label goes below it at the left."""
        b = 4.0 - 0.25 * self.X
        label = label_curve(axes, self.X, [self.A, b], 0, "rising")
        assert label is not None
        box = self.check_beside(axes, label, self.A, b, above=False)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_masked_log_axis(self, axes: matplotlib.axes.Axes) -> None:
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

    def test_none_in_avoid_is_skipped(self, axes: matplotlib.axes.Axes) -> None:
        label = label_curve(axes, self.X, [self.A, self.B], 0, "rising", avoid=[None])
        assert label is not None

    def test_error_leaves_nothing_on_the_axes(self, axes: matplotlib.axes.Axes) -> None:
        not_a_label: Any = "not a label"
        with pytest.raises(AttributeError):
            label_curve(
                axes, self.X, [self.A, self.B], 0, "rising", avoid=[not_a_label]
            )
        assert len(axes.texts) == 0

    @pytest.mark.parametrize(
        "key",
        ["ha", "va", "transform", "rotation_mode", "bbox", "fontdict", "wrap"],
    )
    def test_keywords_set_here_raise(
        self, axes: matplotlib.axes.Axes, key: str
    ) -> None:
        kwargs: dict[str, Any] = {key: None}
        with pytest.raises(ValueError, match="sets or forbids"):
            label_curve(axes, self.X, [self.A, self.B], 0, "t", **kwargs)
        assert len(axes.texts) == 0

    def test_single_curve_goes_leftmost(self, axes: matplotlib.axes.Axes) -> None:
        """With no other curve, every candidate ties, so the label starts at 0."""
        label = label_curve(axes, self.X, [self.A], 0, "alone", side="above")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_all_missing_other_curve_is_ignored(
        self, axes: matplotlib.axes.Axes
    ) -> None:
        ys = [self.A, np.full_like(self.X, np.nan)]
        label = label_curve(axes, self.X, ys, 0, "rising", side="above")
        assert label is not None
        box = self.extent(axes, label)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_identical_curves_give_none(self, axes: matplotlib.axes.Axes) -> None:
        """A label for one of two equal curves is as near the other."""
        assert label_curve(axes, self.X, [self.A, self.A.copy()], 0, "twin") is None
        assert len(axes.texts) == 0

    def test_small_lead_on_parallel_slopes_goes_leftmost(
        self, axes: matplotlib.axes.Axes
    ) -> None:
        """0.3x + 0.08 leads 0.3x by 3.1 pixels, and by 2.3 over the spans
        compared, the other curve's being the wider (3.1 less 0.37 pixels a
        pixel of slope times the 2.2 pixels the spans differ by).  That is
        above the 2.2 needed: the farthest the labeled curve can be from the
        label, less ``gap``.  That farthest distance is the hypotenuse of
        ``gap`` and ``gap`` plus 1.3 pixels, the difference between the
        label's widths on the canvas and in a PDF.

        The curves run on to x = 12, past the axes, so no span is cut short at
        an end of the data, and the leads, equal but for rounding, tie: the
        label starts at 0.

        """
        x = np.linspace(0.0, 12.0, 121)
        own, other = 0.3 * x + 0.08, 0.3 * x
        label = label_curve(axes, x, [own, other], 0, "upper", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other, x=x)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 0.0, atol=1e-6)

    def test_sparse_gap_in_another_curve(self, axes: matplotlib.axes.Axes) -> None:
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

    def test_away_side_judged_by_the_nearest_curve(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    def test_nearest_of_two_curves_ahead_decides(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    @pytest.mark.parametrize(("ahead_px", "placed"), [(23.0, True), (20.6, False)])
    def test_own_peak_at_the_narrower_box_corner(
        self, ahead_px: float, placed: bool
    ) -> None:
        """Another curve must be farther from the label than the labeled curve
        is from the narrower of the label's two boxes.

        Data units here are display pixels: axes 200 by 300 pixels.  The label,
        ten W at 10 points, is 140 pixels wide on the canvas at 100 dpi and
        about 2.6 pixels narrower in a PDF.  It can start only at x = 10.  The
        labeled curve is flat at 100 but for a narrow peak of 30, placed just
        inside the right end of the label's span, ``gap`` past the wider box.
        From the narrower box's corner the peak is about the hypotenuse of
        ``gap`` and ``gap`` plus the 2.6 pixels, 21.6 pixels, where ``gap``
        times the square root of 2 is 19.6.  The other curve is flat, 23 or
        20.6 pixels above the label's far edge; only at 23 is it farther than
        the peak.

        """
        fig = plt.figure(figsize=(4.0, 3.0), dpi=100)
        ax_ = fig.add_axes((0.0, 0.0, 0.5, 1.0))
        ax_.set_xlim(0.0, 200.0)
        ax_.set_ylim(0.0, 300.0)
        text = "W" * 10
        probe = ax_.text(0.0, 0.0, text)
        canvas_box = self.extent(ax_, probe)
        pdf_box = self.pdf_extent(ax_, probe)
        probe.remove()
        assert pdf_box.width < canvas_box.width - 2.0  # the premise above
        gap_px = 10.0 * 100.0 / 72.0
        peak_x = 10.0 + canvas_box.width + gap_px - 0.01
        x = np.array([10.0, peak_x - 2.0, peak_x, peak_x + 2.0, 400.0])
        own = np.array([100.0, 100.0, 130.0, 100.0, 100.0])
        far_edge = 130.0 + gap_px + max(canvas_box.height, pdf_box.height)
        other = np.full(x.shape, far_edge + ahead_px)
        label = label_curve(ax_, x, [own, other], 0, text, side="above", gap=10.0)
        if placed:
            assert label is not None
            self.check_beside(ax_, label, own, other, gap_pts=10.0, x=x)
            np.testing.assert_allclose(
                self.distance(ax_, self.pdf_extent(ax_, label), x, own),
                np.hypot(gap_px, gap_px + canvas_box.width - pdf_box.width - 0.01),
                atol=1e-6,
            )
        else:
            assert label is None

    def test_avoid_as_laid_out_in_a_pdf(self) -> None:
        """A text in ``avoid`` is avoided as a PDF lays it out, too.

        At 72 dpi, "Southern Oscillation Index" is about 2 points wider in a
        PDF than on the canvas.  It sits on the one curve at the left, and the
        label must start clear of both of its boxes, with points 0.1 pixels
        apart to start at.

        """
        _, ax_ = plt.subplots(figsize=(4.0, 3.0), dpi=72)
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        x = np.linspace(0.0, 10.0, 2001)
        own = np.ones_like(x)
        first = ax_.text(0.0, 1.0, "Southern Oscillation Index", va="bottom")
        first_canvas, first_pdf = self.extent(ax_, first), self.pdf_extent(ax_, first)
        assert first_pdf.x1 > first_canvas.x1 + 1.0  # the premise above
        label = label_curve(ax_, x, [own], 0, "a", side="above", avoid=[first])
        assert label is not None
        for box in (self.extent(ax_, label), self.pdf_extent(ax_, label)):
            assert not box.overlaps(first_canvas)
            assert not box.overlaps(first_pdf)

    def test_nearest_of_two_curves_behind_decides(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    def test_slowly_rising_lead_goes_right(self, axes: matplotlib.axes.Axes) -> None:
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

    def test_nearest_of_several_curves_decides(
        self, axes: matplotlib.axes.Axes
    ) -> None:
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

    def test_candidates_only_where_the_labeled_curve_is(
        self, axes: matplotlib.axes.Axes
    ) -> None:
        """b, labeled below, is missing past x = 8, so its label starts at 8.

        The other curve, 3.2 + 0.1x, is above b by more the further right.
        With a gap of 10 points, a label starting at 8.4 would still have b's
        last points within its span, and would win there.

        """
        own = np.where(self.X > 8.0, np.nan, self.B)
        other = 3.2 + 0.1 * self.X
        label = label_curve(axes, self.X, [other, own], 1, "f", side="below", gap=10.0)
        assert label is not None
        box = self.check_beside(axes, label, own, other, gap_pts=10.0, above=False)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 8.0, atol=1e-6)

    @pytest.mark.parametrize("far_first", [True, False])
    def test_avoid_several(self, axes: matplotlib.axes.Axes, far_first: bool) -> None:
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

    def test_fontsize_sets_the_measured_size(self, axes: matplotlib.axes.Axes) -> None:
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

    def test_figure_made_without_pyplot(self) -> None:
        """The canvas of a figure made without pyplot cannot draw, and the
        layout is still settled before measuring.

        With an equal aspect, measuring before the layout put the label 11.3
        pixels from its curve once rendered, where the gap of 10 points is
        13.9 pixels.

        """
        fig = matplotlib.figure.Figure(figsize=(4.0, 3.0), dpi=100)
        ax_ = fig.subplots()
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        ax_.set_aspect("equal")
        label = label_curve(
            ax_, self.X, [self.A, self.B], 0, "rising", side="above", gap=10.0
        )
        assert label is not None
        FigureCanvasAgg(fig)
        self.check_beside(ax_, label, self.A, self.B, gap_pts=10.0)

    def test_axes_on_a_subfigure(self) -> None:
        """Axes on a subfigure are labeled, the layout settled through the
        root figure."""
        fig = plt.figure(figsize=(8.0, 3.0), dpi=100)
        ax_ = fig.add_subfigure(fig.add_gridspec(1, 2)[0, 1]).subplots()
        ax_.set_xlim(0.0, 10.0)
        ax_.set_ylim(-1.0, 5.0)
        label = label_curve(ax_, self.X, [self.A, self.B], 0, "rising", side="above")
        assert label is not None
        self.check_beside(ax_, label, self.A, self.B)

    def test_left_edge_of_the_axes(self, axes: matplotlib.axes.Axes) -> None:
        """With x shown from 2, the lead 1 - 0.05x is largest at x = 2."""
        axes.set_xlim(2.0, 10.0)
        own, other = 1.0 + 0.3 * self.X, 0.35 * self.X
        label = label_curve(axes, self.X, [own, other], 0, "rising", side="above")
        assert label is not None
        box = self.check_beside(axes, label, own, other)
        x_left = axes.transData.inverted().transform((box.x0, box.y0))[0]
        np.testing.assert_allclose(x_left, 2.0, atol=1e-6)

    def test_bottom_edge_of_the_axes(self, axes: matplotlib.axes.Axes) -> None:
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

    @pytest.mark.parametrize("gap", [0.0, -1.0])
    def test_nonpositive_gap_raises(
        self, axes: matplotlib.axes.Axes, gap: float
    ) -> None:
        with pytest.raises(ValueError, match="gap must be positive"):
            label_curve(axes, self.X, [self.A, self.B], 0, "t", gap=gap)

    @pytest.mark.parametrize(
        ("x", "ys", "which", "side", "match"),
        [
            (np.arange(3.0), [np.zeros(3), np.ones(3)], 0, "left", "side must be"),
            (np.arange(3.0), [np.zeros(3), np.ones(3)], 2, "above", "which must"),
            (np.arange(3.0), [np.zeros(3), np.ones(3)], -1, "above", "which must"),
            (np.zeros((3, 2)), [np.zeros(3)], 0, "above", "curve must"),
            (
                np.array([0.0, 2.0, 1.0]),
                [np.zeros(3), np.ones(3)],
                0,
                "above",
                "monoto",
            ),
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

    def test_single_band_with_a_tip_sits_at_start(self) -> None:
        """One band is both the first and the last; it takes ``start``."""
        cmap = plt.get_cmap("viridis")
        colors = sequential_band_colors(cmap, [0.0, 1.0], extend="max")
        assert colors == [cmap(0.1), cmap(1.0)]


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
        eight bands reach it is 4, and its outermost band, -32 to -28, contains
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

    def test_sequential_mostly_at_zero_with_noise_below(self) -> None:
        """The same field with one cell at -1e-15 gets the same bands.

        Its 95th percentile, 0, lies above its minimum of -1e-15, which
        counts as zero, so the percentile still sits at the bottom of the bar
        and the bands cover the extreme.

        """
        field = np.concatenate([np.zeros(1000), [-1e-15, 5.0, 7.0]])
        scale = tight_scale(field, kind="sequential")
        assert scale.levels[0] == -1e-15
        np.testing.assert_array_equal(scale.levels[1:], np.arange(1, 15) * 0.5)
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
        """A masked array's masked cells, here set to 1e20, are left out."""
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

    def test_sequential_top_on_a_grid_multiple(self) -> None:
        """0.07 / 0.01 is 7.000000000000001, which must not refuse seven bands of 0.01."""
        field = np.linspace(0.0, 0.07, 71)
        scale = tight_scale(field, kind="sequential", num_bands=7)
        assert scale.step == 0.01 and scale.levels.size == 8

    def test_sequential_one_band(self) -> None:
        """4.95 to 5.05 in at most one band: no width up to 1 does it, 2 does."""
        scale = tight_scale(np.array([4.95, 5.05]), kind="sequential", num_bands=1)
        np.testing.assert_array_equal(scale.levels, [4.0, 6.0])

    def test_sequential_noise_below_zero(self) -> None:
        """A minimum of -1e-15, regridding noise, is zero to within rounding."""
        scale = tight_scale(np.array([-1e-15, 3.0]), kind="sequential")
        assert scale.levels[0] == -1e-15 and scale.extend == "neither"

    def test_sequential_noise_below_zero_adds_no_band(self) -> None:
        """A minimum of -2.5e-9, within rounding of 3.0, widens the first band.

        It is below zero by more than the grid's snapping tolerance, 1e-9 of
        the width of 0.2, but within the 1e-9 of 3.0 that a sequential field
        may be negative by.  So the bar is fifteen bands of 0.2 from zero,
        the first starting at -2.5e-9, and zero is labeled.

        """
        scale = tight_scale(np.r_[-2.5e-9, np.linspace(0, 3, 301)], kind="sequential")
        assert scale.levels[0] == -2.5e-9
        expected = [float(f"{v:.12g}") for v in np.arange(1, 16) * 0.2]
        assert scale.levels[1:].tolist() == expected
        assert scale.extend == "neither"
        assert colorbar_ticks(scale.levels)[0] == 0.0

    def test_mostly_near_zero_field_covers_only_those_values(self) -> None:
        """The precipitation-like example in the docstring.

        With 96% of the values from 0 to 0.01, the 95th percentile falls among
        them, so the bar spans 0 to 0.01 and the 40 values from 0.5 to 40
        are all in its tip.

        """
        field = np.r_[np.linspace(0.0, 0.01, 960), np.linspace(0.5, 40.0, 40)]
        scale = tight_scale(field, kind="sequential")
        np.testing.assert_allclose(scale.levels, np.arange(11) * 0.001, rtol=1e-12)
        assert scale.extend == "max"
        assert np.sum(field > scale.levels[-1]) == 40

    @pytest.mark.parametrize(
        ("field", "kind", "end"),
        [
            ([-1.0, 1 / 3], "diverging", -1),
            ([2 / 3, 3.0], "sequential", 0),
        ],
    )
    def test_step_without_a_short_decimal(
        self, field: list[float], kind: str, end: int
    ) -> None:
        """A width of 1/3 leaves levels unrounded, so an extreme on the grid ends it.

        Rounded to six significant figures of the width, 1/3 would become
        0.3333333, short of the extreme by far more than rounding, giving a
        spurious arrow tip, which ``sequential_band_colors`` rejects at the
        lower end of a sequential bar.

        """
        scale = tight_scale(np.array(field), kind=kind, step=1 / 3)
        assert scale.levels[end] == field[end]
        assert scale.extend == "neither"

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

    def test_extreme_past_seven_widths_by_rounding_takes_the_narrower(self) -> None:
        """An extreme past seven bands of 4 by rounding alone counts as at them.

        Taken as past them, it would get an eighth band of 4, from -32 to -28,
        containing no value visibly different from -28.

        """
        scale = tight_scale(np.array([np.nextafter(-28.0, -np.inf), 1.0]))
        np.testing.assert_array_equal(scale.levels, np.arange(-8, 2) * 3.0)
        assert scale.extend == "min"

    def test_bands_per_side_not_a_power_of_two(self) -> None:
        """Six bands of 0.3 reach 1.8 exactly, which a test without a tolerance refuses."""
        field = np.concatenate([np.full(99, -1.8), [-5.0, 0.5]])
        scale = tight_scale(field, bands_per_side=6)
        assert scale.step == 0.3

    @pytest.mark.parametrize("seed", range(20))
    def test_rules_hold_on_random_fields(self, seed: int) -> None:
        """The rules as invariants, over shifted and heavy-tailed random fields.

        The longer side has exactly eight bands; an asymmetric bar's shorter
        side never tips; and each end of the bar either tips or has data in
        its outermost band, symmetric bars included, except the end at zero
        of a one-signed field.  The shift of up to 3 standard deviations makes
        some fields one-signed.

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
        else:
            assert tips_low or -lo > (n_below - 1) * step
            assert tips_high or hi > (n_above - 1) * step

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
        "field",
        [
            [288.0, 288.00000000000006],
            [271.35, 271.35 + 1e-13],
            [1.0, 1.0 + 2e-16],
        ],
    )
    def test_constant_but_for_rounding_raises(self, field: list[float]) -> None:
        """Values a few units in the last place apart give no distinct levels."""
        with pytest.raises(ValueError, match="differ only by rounding"):
            tight_scale(np.array(field), kind="sequential")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bands_per_side": 0}, "at least 1"),
            ({"num_bands": 0}, "at least 1"),
            ({"bands_per_side": 2.5}, "integers"),
            ({"num_bands": 2.5}, "integers"),
            ({"quantile": 0.0}, "quantile must"),
            ({"quantile": 1.5}, "quantile must"),
            ({"tip_ratio": 0.0}, "tip_ratio must"),
            ({"step": np.nan}, "step must be positive"),
            ({"step": np.inf}, "step must be positive and finite"),
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

    def test_log_spaced_levels_label_the_smallest_as_zero(self) -> None:
        """The limitation the docstring notes, for levels from 0.001 to 1e6.

        1e-9 times the largest level, 1e6, is 0.001, so the level 0.001
        counts as zero and is labeled 0.

        """
        assert colorbar_ticks(10.0 ** np.arange(-3, 7)) == [0.0, 1.0, 1e3, 1e6]

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

    def test_levels_without_a_short_decimal_are_kept(self) -> None:
        """Thirds are labeled where they lie, on the boundaries themselves.

        Rounded to six significant figures, 2/3 would become 0.6666667, off
        the boundary it labels and past the top of the bar.

        """
        levels = np.arange(-3, 3) / 3.0
        assert colorbar_ticks(levels, every=1) == levels.tolist()

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
