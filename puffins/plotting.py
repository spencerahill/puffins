#! /usr/bin/env python
"""Helper functions for creating plots."""

import copy
import io
import os.path
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, cast

import numpy as np
import xarray as xr
from faceted import faceted as fac_faceted
from faceted import faceted_ax as fac_ax
from matplotlib import colors, gridspec, ticker
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

from ._typing import ArrayLike
from .names import LAT_STR
from .nb_utils import sindeg
from .stats import detrend, dt_std_anom, lin_regress, trend

_DEGR = r"$^\circ$"
_DEGR_S = _DEGR + "S"
_DEGR_N = _DEGR + "N"
GRAY = "0.4"

PlotArr = namedtuple("PlotArr", ["func", "label", "plot_kwargs"])

plt_rc_params_custom = {
    "axes.edgecolor": GRAY,  # Make axis spines gray.
    "axes.labelcolor": GRAY,  # Make axis labels gray.
    "axes.spines.top": False,  # Turn off top spine in plots.
    "axes.spines.right": False,  # Turn off right spine in plots.
    "figure.dpi": 100,  # Make inline figures larger in Jupyter notebooks.
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "DejaVu Sans",
    ],  # Use Helvetica font if it's available.
    "legend.frameon": False,  # Turn off box around legend.
    "legend.handlelength": 1.0,  # Make legend symbols smaller.
    "legend.labelcolor": "linecolor",  # Make legend text color match that element
    "mathtext.fontset": "cm",  # Use serifed font in equations.
    "pdf.fonttype": 42,  # Bug workaround: https://stackoverflow.com/a/60384073
    "text.color": GRAY,  # Make text gray.
    "xtick.color": GRAY,  # Make xticks gray.
    "ytick.color": GRAY,  # Make yticks gray.
}


def _gca_if_ax_none(ax):
    """Get the currently active matplotlib Axes if one isn't provided."""
    if ax is None:
        return plt.gca()
    return ax


def _left_bottom_spines_only(ax=None, displace=False):
    """Don't plot top or right border."""
    ax = _gca_if_ax_none(ax)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if displace:
        ax.spines["left"].set_position(("outward", 20))
        ax.spines["bottom"].set_position(("outward", 20))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


# Format axes for plots that are functions of latitude.
def sinlat_xaxis(
    ax=None, start_lat=-90, end_lat=90, do_ticklabels=False, degr_symbol=False
):
    """Make the x-axis be in sin of latitude."""
    ax = _gca_if_ax_none(ax)
    ax.set_xlim([sindeg(start_lat), sindeg(end_lat)])
    if start_lat == 0 and end_lat == 90:
        ax.set_xticks(sindeg([0, 30, 60, 90]))
        ax.set_xticks(sindeg([10, 20, 40, 50, 70, 80]), minor=True)
        if do_ticklabels:
            if degr_symbol:
                ax.set_xticklabels(["EQ", r"30$^\circ$" r"60$^\circ$", r"90$^\circ$"])
            else:
                ax.set_xticklabels(["EQ", r"30N" r"60N", r"90N"])
    elif start_lat == -90 and end_lat == 90:
        ax.set_xticks(sindeg([-90, -60, -30, 0, 30, 60, 90]))
        minorticks = [-80, -70, -50, -40, -20, -10, 10, 20, 40, 50, 70, 80]
        ax.set_xticks(sindeg(minorticks), minor=True)
        if do_ticklabels:
            if degr_symbol:
                ax.set_xticklabels(
                    [
                        "90" + _DEGR_S,
                        " ",
                        "30" + _DEGR_S,
                        "EQ",
                        "30" + _DEGR_N,
                        " ",
                        "90" + _DEGR_N,
                    ]
                )
            else:
                ax.set_xticklabels(["90S", "", "30S", "EQ", "30N", "", "90N"])


def lat_xaxis(ax=None, start_lat=-90, end_lat=90, degr_symbol=False, **kwargs):
    """Make the x-axis be latitude."""
    ax = _gca_if_ax_none(ax)
    ax.set_xlim([start_lat, end_lat])

    if start_lat == 0 and end_lat == 90:
        ticks = [0, 30, 60, 90]
        minor_ticks = [10, 20, 40, 50, 70, 80]
        if degr_symbol:
            ticklabels = ["EQ", "30" + _DEGR, "60" + _DEGR, "90" + _DEGR]
        else:
            ticklabels = ["EQ", "30N", "60N", "90N"]
    elif start_lat == -90 and end_lat == 90:
        ticks = [-90, -60, -30, 0, 30, 60, 90]
        minor_ticks = [-80, -70, -50, -40, -20, -10, 10, 20, 40, 50, 70, 80]
        if degr_symbol:
            ticklabels = [
                f"90{_DEGR_S}",
                f"60{_DEGR_S}",
                f"30{_DEGR_S}",
                "EQ",
                f"30{_DEGR_N}",
                f"60{_DEGR_N}",
                f"90{_DEGR_N}",
            ]
        else:
            ticklabels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]
    elif start_lat == -30 and end_lat == 30:
        ticks = [-30, -20, -10, 0, 10, 20, 30]
        minor_ticks = [-25, -15, -5, 5, 15, 25]
        if degr_symbol:
            ticklabels = [
                f"30{_DEGR_S}",
                f"20{_DEGR_S}",
                f"10{_DEGR_S}",
                "EQ",
                f"10{_DEGR_N}",
                f"20{_DEGR_N}",
                f"30{_DEGR_N}",
            ]
        else:
            ticklabels = ["30S", "20S", "10S", "EQ", "10N", "20N", "30N"]

    elif start_lat == -45 and end_lat == 45:
        ticks = [-45, -30, -15, 0, 15, 30, 45]
        minor_ticks = [-40, -35, -25, -20, -10, -5, 5, 10, 20, 25, 35, 40]
        if degr_symbol:
            ticklabels = [
                f"45{_DEGR_S}",
                f"30{_DEGR_S}",
                f"15{_DEGR_S}",
                "EQ",
                f"15{_DEGR_N}",
                f"30{_DEGR_N}",
                f"45{_DEGR_N}",
            ]
        else:
            ticklabels = ["45S", "30S", "15S", "EQ", "15N", "30N", "45N"]
    elif start_lat == -60 and end_lat == 60:
        ticks = [-60, -30, 0, 30, 60]
        minor_ticks = [-50, -40, -20, -10, 10, 20, 40, 50]
        if degr_symbol:
            ticklabels = [
                f"60{_DEGR_S}",
                f"30{_DEGR_S}",
                "EQ",
                f"30{_DEGR_N}",
                f"60{_DEGR_N}",
            ]
        else:
            ticklabels = ["60S", "30S", "EQ", "30N", "60N"]
    else:
        ticks = np.arange(start_lat, end_lat + 1, 10)
        minor_ticks = None
        ticklabels = None
    ax.set_xticks(ticks)
    if minor_ticks is not None:
        ax.set_xticks(minor_ticks, minor=True)
    if ticklabels is not None:
        ax.set_xticklabels(ticklabels, **kwargs)
    ax.set_xlabel("")


def lat_yaxis(ax=None, start_lat=-90, end_lat=90, degr_symbol=False, **kwargs):
    """Make the y-axis be latitude."""
    ax = _gca_if_ax_none(ax)
    ax.set_ylim([start_lat, end_lat])

    if start_lat == 0 and end_lat == 90:
        ticks = [0, 30, 60, 90]
        minor_ticks = [10, 20, 40, 50, 70, 80]
        if degr_symbol:
            ticklabels = ["EQ", "30" + _DEGR, "60" + _DEGR, "90" + _DEGR]
        else:
            ticklabels = ["EQ", "30N", "60N", "90N"]
    elif start_lat == -90 and end_lat == 90:
        ticks = [-90, -60, -30, 0, 30, 60, 90]
        minor_ticks = [-80, -70, -50, -40, -20, -10, 10, 20, 40, 50, 70, 80]
        if degr_symbol:
            ticklabels = [
                f"90{_DEGR_S}",
                f"60{_DEGR_S}",
                f"30{_DEGR_S}",
                "EQ",
                f"30{_DEGR_N}",
                f"60{_DEGR_N}",
                f"90{_DEGR_N}",
            ]
        else:
            ticklabels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]
    elif start_lat == -30 and end_lat == 30:
        ticks = [-30, -20, -10, 0, 10, 20, 30]
        minor_ticks = [-25, -15, -5, 5, 15, 25]
        if degr_symbol:
            ticklabels = [
                f"30{_DEGR_S}",
                f"20{_DEGR_S}",
                f"10{_DEGR_S}",
                "EQ",
                f"10{_DEGR_N}",
                f"20{_DEGR_N}",
                f"30{_DEGR_N}",
            ]
        else:
            ticklabels = ["30S", "20S", "10S", "EQ", "10N", "20N", "30N"]

    elif start_lat == -45 and end_lat == 45:
        ticks = [-45, -30, -15, 0, 15, 30, 45]
        minor_ticks = [-40, -35, -25, -20, -10, -5, 5, 10, 20, 25, 35, 40]
        if degr_symbol:
            ticklabels = [
                f"45{_DEGR_S}",
                f"30{_DEGR_S}",
                f"15{_DEGR_S}",
                "EQ",
                f"15{_DEGR_N}",
                f"30{_DEGR_N}",
                f"45{_DEGR_N}",
            ]
        else:
            ticklabels = ["45S", "30S", "15S", "EQ", "15N", "30N", "45N"]
    elif start_lat == -60 and end_lat == 60:
        ticks = [-60, -30, 0, 30, 60]
        minor_ticks = [-50, -40, -20, -10, 10, 20, 40, 50]
        if degr_symbol:
            ticklabels = [
                f"60{_DEGR_S}",
                f"30{_DEGR_S}",
                "EQ",
                f"30{_DEGR_N}",
                f"60{_DEGR_N}",
            ]
        else:
            ticklabels = ["60S", "30S", "EQ", "30N", "60N"]
    else:
        ticks = np.arange(start_lat, end_lat + 1, 10)
        minor_ticks = None
        ticklabels = None
    ax.set_yticks(ticks)
    if minor_ticks is not None:
        ax.set_yticks(minor_ticks, minor=True)
    if ticklabels is not None:
        ax.set_yticklabels(ticklabels, **kwargs)
    ax.set_ylabel("")


def sinlat_yaxis(
    ax=None, start_lat=-90, end_lat=90, do_ticklabels=False, degr_symbol=False
):
    """Make the x-axis be in sin of latitude."""
    ax = _gca_if_ax_none(ax)
    ax.set_ylim([sindeg(start_lat), sindeg(end_lat)])
    if start_lat == 0 and end_lat == 90:
        ax.set_yticks(sindeg([0, 30, 60, 90]))
        ax.set_yticks(sindeg([10, 20, 40, 50, 70, 80]), minor=True)
        if do_ticklabels:
            if degr_symbol:
                ax.set_yticklabels(["EQ", r"30$^\circ$" r"60$^\circ$", r"90$^\circ$"])
            else:
                ax.set_yticklabels(["EQ", r"30N" r"60N", r"90N"])
    elif start_lat == -90 and end_lat == 90:
        ax.set_yticks(sindeg([-90, -60, -30, 0, 30, 60, 90]))
        minorticks = [-80, -70, -50, -40, -20, -10, 10, 20, 40, 50, 70, 80]
        ax.set_yticks(sindeg(minorticks), minor=True)
        if do_ticklabels:
            if degr_symbol:
                ax.set_yticklabels(
                    [
                        "90" + _DEGR_S,
                        " ",
                        "30" + _DEGR_S,
                        "EQ",
                        "30" + _DEGR_N,
                        " ",
                        "90" + _DEGR_N,
                    ]
                )
            else:
                ax.set_yticklabels(["90S", "", "30S", "EQ", "30N", "", "90N"])


def plot_lat_1d(
    arr,
    start_lat=-90,
    end_lat=90,
    sinlat=False,
    ax=None,
    lat_str=LAT_STR,
    ax_labels=False,
    **plot_kwargs,
):
    """Plot of the given array as a function of latitude."""
    ax = _gca_if_ax_none(ax)
    arr_plot = arr.where((arr[lat_str] > start_lat) & (arr[lat_str] < end_lat))
    if sinlat:
        lat = sindeg(arr_plot[lat_str])
        sinlat_xaxis(ax, start_lat=start_lat, end_lat=end_lat)
    else:
        lat = arr_plot[lat_str]
        lat_xaxis(ax, start_lat=start_lat, end_lat=end_lat)

    handle = ax.plot(lat, arr_plot, **plot_kwargs)[0]

    _left_bottom_spines_only(ax, displace=False)

    if ax_labels:
        ax.set_xlabel(r"Latitude [$^\circ$]")
        if arr.name:
            ax.set_ylabel(arr.name)

    return handle


# Annual cycle plots
def wrap_ann_cyc(arr, dim="month"):
    """Append extra Jan and Dec. values to the array.

    So that contour etc. plots of the annual cycle show
    January and December as full width rather than half.

    """
    arr_wrapped = arr.pad({dim: 1}, mode="wrap")
    arr_wrapped.coords["month"] = range(14)
    return arr_wrapped


def ann_cyc_xaxis(ax=None, wrap=True):
    """Format the x axis to be the calendar months."""
    ax = _gca_if_ax_none(ax)
    if wrap:
        ax.set_xlim(0.5, 12.5)
    else:
        ax.set_xlim(1, 12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels("JFMAMJJASOND")
    ax.set_xlabel("")
    return ax


def plot_ann_cyc(arr, dim="month", ax=None, plot_type=None, **kwargs):
    """Plot annual cycle along x axis, with wrapping."""
    ax = _gca_if_ax_none(ax)
    plot_arr = wrap_ann_cyc(arr, dim=dim)
    plot_funcs = {
        None: plot_arr.plot,
        "contour": plot_arr.plot.contour,
        "contourf": plot_arr.plot.contourf,
    }
    plot_func = plot_funcs[plot_type]
    handle = plot_func(ax=ax, x=dim, **kwargs)
    ann_cyc_xaxis(ax=ax, wrap=True)
    return handle


# Convenience functions for generating figure and axes instances
def faceted(*args, width=4, aspect=0.618, **kwargs):
    """Wrapper to faceted.faceted w/ a default aspect ratio."""
    return fac_faceted(*args, width=width, aspect=aspect, **kwargs)


def faceted_ax(*args, width=4, aspect=0.618, **kwargs):
    """Wrapper to faceted.faceted_ax w/ a default aspect ratio."""
    return fac_ax(*args, width=width, aspect=aspect, **kwargs)


def fig_ax(width: float = 3.75, aspect: float = 1.618) -> tuple[Figure, Axes]:
    """
    Create plt.Figure with single subplot and specified width and aspect ratio.

    Parameters:
    - width (float): Width of the figure in inches.  Default 3.75 inches.
    - aspect (float): Ratio of width to height.  Default golden ratio (1.618).

    Returns:
    - fig (Figure): The created figure.
    - ax (Axes): The subplot added to the figure.

    """
    height = width / aspect
    return plt.subplots(figsize=(width, height), constrained_layout=True)


# Panel labeling.
def panel_label(
    panel_num=None,
    ax=None,
    extra_text=None,
    x=0.01,
    y=0.99,
    uppercase=False,
    brackets=True,
    bold=False,
    **text_kwargs,
):
    """Annotate an Axes, or an iterable of Axes, with panel labels.

    With `panel_num` of None, `ax` is treated as an iterable and each element
    is labeled in turn.  `uppercase` gives "A, B, C..." rather than
    "a, b, c..."; `brackets` surrounds the letter in parentheses; `bold`
    renders the label in bold, including any `extra_text`.

    `bold` only supplies a default, so an explicit `fontweight` overrides it.

    """
    ax = _gca_if_ax_none(ax)

    if "va" in text_kwargs:
        vert_align = text_kwargs.pop("va")
    elif "verticalalignment" in text_kwargs:
        vert_align = text_kwargs.pop("verticalalignment")
    else:
        vert_align = "top"
    text_kwargs["va"] = vert_align

    if panel_num is None:
        for n, ax_ in enumerate(ax):
            panel_label(
                n,
                ax=ax_,
                extra_text=extra_text,
                x=x,
                y=y,
                uppercase=uppercase,
                brackets=brackets,
                bold=bold,
                **text_kwargs,
            )
        return
    letters = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if uppercase else "abcdefghijklmnopqrstuvwxyz"
    )
    if not 0 <= panel_num < len(letters):
        raise ValueError(
            f"panel_num must be between 0 and {len(letters) - 1}; got {panel_num}"
        )
    letter = letters[panel_num]
    label = f"({letter})" if brackets else letter
    if extra_text is not None:
        label += f" {extra_text}"
    if bold:
        text_kwargs.setdefault("fontweight", "bold")
    ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)


# Curve labeling.
# Keywords label_curve sets itself, or that would move the drawn text off the
# box it checks: fontdict can carry any of the others, and wrap sets the text's
# width by where it is drawn.
_LABEL_KWARGS_SET_HERE = (
    "ha",
    "va",
    "horizontalalignment",
    "verticalalignment",
    "transform",
    "rotation_mode",
    "bbox",
    "fontdict",
    "wrap",
)
# Display pixels by which one candidate must beat another to be preferred, and
# within which the nearest curves behind and ahead are equally near, so that
# distances equal but for rounding tie.
_LABEL_TIE_PX = 1e-6


def _pdf_renderer(fig: Figure) -> RendererPdf:
    """A renderer that lays text out as ``fig`` saved to PDF would."""
    width, height = fig.get_size_inches() * 72.0
    return RendererPdf(PdfFile(io.BytesIO()), 72, height, width)


def _pdf_extent(text: Text, renderer: RendererPdf) -> Bbox:
    """The text's box as a saved PDF lays it out, in the figure's display
    pixels.

    A PDF lays text out at 72 dpi from unhinted glyphs, so its box can be
    wider or narrower than on the figure's canvas.  The figure is set to 72
    dpi for the measurement, as ``savefig`` sets it for a PDF, and its
    canvas is detached from its window meanwhile, as ``savefig`` detaches
    it: each change of dpi otherwise resizes the window, which on a screen
    backend can change the figure's size.  Measured on a copy, so that the
    text keeps the renderer it caches for the canvas.  An Annotation is
    measured without its arrow, which the copy shares with the original and
    which laying it out for a PDF would move.
    """
    fig = cast(Figure, text.get_figure(root=True))
    canvas, dpi = fig.canvas, fig.dpi
    manager = canvas.manager
    measured = copy.copy(text)
    if isinstance(measured, Annotation):
        measured.arrowprops = None
        measured.arrow_patch = None
    try:
        canvas.manager = None
        fig.dpi = 72.0
        # The dpi argument of Text.get_window_extent sets the figure's dpi
        # with the window attached, and an Annotation's takes no dpi.
        box = measured.get_window_extent(renderer)
    finally:
        # The dpi first, with the window still detached; the window is
        # reattached even if restoring the dpi raises.
        try:
            fig.dpi = dpi
        finally:
            canvas.manager = manager
    return Bbox(box.get_points() * dpi / 72.0)


def _drawn(text: Text) -> bool:
    """Whether the text is drawn: not hidden, and for an Annotation, not
    clipped for having its point outside the axes, which draws neither its
    text nor its arrow."""
    if not text.get_visible():
        return False
    # Annotation.draw makes the same check, with a method its stubs omit.
    check_xy = getattr(text, "_check_xy", None)
    return check_xy is None or bool(check_xy())


def _drawn_points(
    px: np.ndarray, py: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Curves' display points from left to right, NaN where nothing is drawn.

    Matplotlib joins each point to its neighbors in the arrays, so the points
    keep that order, reversed when x decreases; an x that turns back on
    itself raises a ``ValueError``.  A point with a NaN x or y is drawn as a
    gap, and so is a point between two such points, since a segment needs
    two points: the heights there are NaN.  A point with a NaN x takes the x
    of the point before it, so that the x returned is sorted.
    """
    known_x = np.isfinite(px)
    steps = np.diff(px[known_x])
    if (steps < 0).any():
        if (steps > 0).any():
            raise ValueError("x must be monotonic, increasing or decreasing")
        px, py, known_x = px[::-1], [h[::-1] for h in py], known_x[::-1]
    drawn = []
    for heights in py:
        finite = known_x & np.isfinite(heights)
        joined = np.zeros_like(finite)
        joined[1:] |= finite[:-1]
        joined[:-1] |= finite[1:]
        drawn.append(np.where(finite & joined, heights, np.nan))
    if known_x.any():
        px = np.fmax.accumulate(px)
        px = np.where(np.isnan(px), px[known_x][0], px)
    return px, drawn


def _drawn_range(
    px: np.ndarray, heights: np.ndarray, left: float, right: float
) -> tuple[float, float] | None:
    """Smallest and largest of a curve as drawn, over display x from ``left``
    to ``right``.

    ``px`` is sorted, and ``heights`` is NaN wherever no segment is drawn, as
    ``_drawn_points`` gives them.  The curve's finite points in the span
    count, and so do its values interpolated at the span's two ends, but only
    along a segment whose two points are both finite: a NaN gap, and the
    stretch past either end of the curve, are drawn blank and count as no
    curve.  None when no drawn part of the curve lies in the span.
    """
    i_lo = int(np.searchsorted(px, left, side="left"))
    i_hi = int(np.searchsorted(px, right, side="right"))
    inside = heights[i_lo:i_hi]
    inside = inside[np.isfinite(inside)]
    vals = [float(inside.min()), float(inside.max())] if inside.size else []
    for edge, i in ((left, i_lo), (right, i_hi)):
        if 0 < i < px.size and np.isfinite(heights[i - 1]) and np.isfinite(heights[i]):
            vals.append(
                float(np.interp(edge, px[i - 1 : i + 1], heights[i - 1 : i + 1]))
            )
    if not vals:
        return None
    return min(vals), max(vals)


def label_curve(
    ax: Axes,
    x: ArrayLike | Sequence[float],
    ys: Sequence[ArrayLike | Sequence[float]],
    which: int,
    text: str,
    *,
    side: str = "auto",
    gap: float = 2.0,
    avoid: Sequence[Text | None] = (),
    **text_kwargs: Any,
) -> Text | None:
    """Label one of several curves directly, beside it, in place of a legend.

    Implements Spencer Hill's rules for direct labels (2026-07-22 and
    2026-09-09): a label sits beside its curve and never on it, nearer to it
    than to any other curve, at the x where its curve stands farthest from
    the nearest other curve, on the side away from that curve.

    The label's size is measured twice: as drawn on the figure's canvas, such
    as the screen or a PNG file, and as matplotlib lays it out in a saved PDF.
    The two differ, since a PDF lays text out at 72 dpi from unhinted
    glyphs, and the label is checked as the larger of the two, so that what
    follows applies to both.

    Each candidate puts the label's left edge at one of the labeled curve's
    points, on one side of it, with its near edge ``gap`` points beyond the
    labeled curve's extreme over the label's width plus ``gap`` either side.
    So the labeled curve is at least ``gap`` from the label, and at most the
    hypotenuse of ``gap`` and ``gap`` plus the difference between the two
    widths, at a corner of the narrower box.  Every other curve must be
    farther than that: over the label's width plus that distance either
    side, it must lie wholly behind the labeled curve's extreme, or wholly
    beyond the label's far edge, by more than that distance from the label.
    So the label is clear of every curve and nearer the labeled one.  Of the
    candidates lying wholly inside the axes and overlapping no text or arrow
    in ``avoid``, the one where the labeled curve's extreme stands farthest
    from the nearest other curve is taken, the leftmost on a tie, and above
    on a tie at one point.  With ``side="auto"``, candidates on the side away
    from the nearest other curve come first: one on the side toward it is
    taken only when no candidate on the side away from it fits.

    A curve counts only where it is drawn: along segments between two finite
    points, so neither across a NaN gap, nor at a lone point between two
    missing ones, nor past either end of the curve.  Matplotlib joins the
    points in the order given, so ``x`` must be monotonic, increasing or
    decreasing, for that order to run along the x axis.
    Curves are taken as straight segments through their points, centered on
    the line: markers, a stepped drawstyle and the line's own width are not
    counted, so pass a larger ``gap`` for them.

    Distances are measured in display coordinates after a draw, so they apply
    whatever the axis scales.  Other vector formats than PDF are not measured.
    The label is anchored in
    data coordinates while its size and its gap are in points, so draw every
    curve and fix the axis limits, the figure size and the layout first: a
    later change to any of them can bring the label nearer a curve or push it
    outside the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes holding the curves.
    x : array-like
        The curves' shared x values, one-dimensional and monotonic, as
        plotted; dates are converted by the x axis's own units.  NaN or a
        mask marks a missing point.
    ys : sequence of array-like
        Every curve on the axes that the label must keep clear of, each with
        one value per ``x``; NaN or a mask marks a missing point.
    which : int
        Index in ``ys`` of the curve to label.
    text : str
        The label.
    side : {"auto", "above", "below"}
        Which side of its curve the label goes on, as seen on the screen.
        ``"auto"`` tries both, preferring the side away from the nearest
        other curve as the rules specify.  With a side given, the x is
        chosen by the distance to the nearest other curve alone.
    gap : float
        Distance, in points, from the labeled curve's line to the label's
        near edge; positive.
    avoid : sequence of matplotlib.text.Text or None
        Labels already placed, which this one must not overlap on the canvas
        or in a PDF.  An Annotation's arrow counts as well, as the path the
        canvas draws it along, widened by half its line width either side.
        An Annotation positioned in pixels, by ``"figure pixels"``,
        ``"axes pixels"`` or ``"offset pixels"``, has its arrow elsewhere in
        a PDF, where the label can overlap it.  Measuring an Annotation
        positioned relative to another artist leaves that artist's own
        measurement wrong until the next draw, so list that artist before
        the Annotation.  Texts not drawn, being hidden or clipped, and None
        entries, as from earlier calls that placed nothing, are skipped.
    **text_kwargs
        Passed to ``Axes.text``, such as ``color`` and ``fontsize``.  The
        alignment, the transform, the rotation mode and a bounding-box patch
        are set here or would move the drawn text off the box checked, and
        raise a ``ValueError`` if passed; so do ``fontdict``, which can carry
        any of them, and ``wrap``, which sets the text's width by where it
        is drawn.

    Returns
    -------
    matplotlib.text.Text or None
        The label, or None when no position satisfies the rules, in which case
        nothing is added to the axes; nothing is left on the axes if an
        error is raised either.

    """
    if side not in ("auto", "above", "below"):
        raise ValueError(f"side must be 'auto', 'above' or 'below'; got {side!r}")
    if not gap > 0:
        raise ValueError(f"gap must be positive; got {gap}")
    blocked = sorted(set(text_kwargs) & set(_LABEL_KWARGS_SET_HERE))
    if blocked:
        raise ValueError(f"label_curve sets or forbids {blocked}; do not pass them")
    # Masked values are drawn as gaps, as NaN is.
    curves: list[np.ndarray] = [
        np.ma.filled(np.ma.asarray(y, dtype=float), np.nan) for y in ys
    ]
    x_masked: np.ma.MaskedArray[Any, Any] = np.ma.asarray(x)
    x_vals = np.asarray(ax.xaxis.convert_units(x_masked.data), dtype=float)
    if x_vals.ndim != 1 or any(c.shape != x_vals.shape for c in curves):
        raise ValueError("x must be one-dimensional and each curve must match it")
    if not 0 <= which < len(curves):
        raise ValueError(f"which must index ys, of length {len(curves)}; got {which}")
    x_vals = np.where(np.ma.getmaskarray(x_masked), np.nan, x_vals)
    # The root figure, which a subfigure's axes reach through their subfigure;
    # an Axes always belongs to one.
    fig = cast(Figure, ax.get_figure(root=True))
    # Settles the layout, such as an equal aspect, on any canvas: the canvas
    # of a figure made without pyplot draws nothing.
    fig.draw_without_rendering()
    # Measured at the axes' center, a point every axis scale can place.
    label = ax.text(
        0.5, 0.5, text, transform=ax.transAxes, ha="left", va="bottom", **text_kwargs
    )
    placed = False
    try:
        pdf_renderer = _pdf_renderer(fig)
        on_canvas = label.get_window_extent()
        in_pdf = _pdf_extent(label, pdf_renderer)
        # The box checked covers the label both as drawn on the canvas and as
        # laid out in a PDF.  Both share the left edge and the near edge.
        width = max(on_canvas.width, in_pdf.width)
        height = max(on_canvas.height, in_pdf.height)
        gap_px = gap * fig.dpi / 72.0
        # The farthest the labeled curve can be from the label: at a corner of
        # the narrower of the two, with its extreme up to gap past the wider.
        reach = float(np.hypot(gap_px, gap_px + abs(on_canvas.width - in_pdf.width)))
        avoided = [t for t in avoid if t is not None and _drawn(t)]
        # Each text's own box, without an Annotation's arrow, which
        # Text.get_window_extent leaves out.
        others_placed = [
            Bbox.union([Text.get_window_extent(t), _pdf_extent(t, pdf_renderer)])
            for t in avoided
        ]
        # Each arrow as the path the canvas draws, with half its line width,
        # by which a label must clear the path.
        arrows = [
            (
                t.arrow_patch.get_transform().transform_path(t.arrow_patch.get_path()),
                t.arrow_patch.get_linewidth() * fig.dpi / 72.0 / 2.0,
            )
            for t in avoided
            if isinstance(t, Annotation) and t.arrow_patch is not None
        ]
        # Display x from x alone: a NaN y makes the transform's x NaN as well,
        # which would drop the other curves' points where the labeled one is
        # missing.
        y_ref = np.full_like(x_vals, float(np.mean(ax.get_ylim())))
        px, py = _drawn_points(
            ax.transData.transform(np.column_stack([x_vals, y_ref]))[:, 0],
            [
                ax.transData.transform(np.column_stack([x_vals, c]))[:, 1]
                for c in curves
            ],
        )
        frame = ax.bbox
        best_key: tuple[bool, float] | None = None
        # The label's left edge, its near edge, and its side.
        best_place: tuple[float, float, str] | None = None
        for left in px[np.isfinite(py[which])]:
            right = left + width
            if left < frame.x0 or right > frame.x1:
                continue
            # Each curve's range is found once for both sides.  Never None: the
            # candidate's own point lies in its span.
            own_lo, own_hi = cast(
                tuple[float, float],
                _drawn_range(px, py[which], left - gap_px, right + gap_px),
            )
            ranges = [
                _drawn_range(px, curve_py, left - reach, right + reach)
                for k, curve_py in enumerate(py)
                if k != which
            ]
            drawn = [r for r in ranges if r is not None]
            for this_side in ("above", "below") if side == "auto" else (side,):
                # Heights are measured toward the label, so below the labeled
                # curve they are negated, which negates and swaps each range.
                if this_side == "above":
                    own, others = own_hi, drawn
                else:
                    own, others = -own_lo, [(-hi, -lo) for lo, hi in drawn]
                near_edge = own + gap_px
                far_edge = near_edge + height
                behind, ahead = [], []
                for lo, hi in others:
                    if hi < near_edge - reach:
                        behind.append(own - hi)
                    elif lo > far_edge + reach:
                        ahead.append(lo - own)
                    else:
                        break
                else:
                    if this_side == "above":
                        y_lo, y_hi = near_edge, far_edge
                    else:
                        y_lo, y_hi = -far_edge, -near_edge
                    if y_lo < frame.y0 or y_hi > frame.y1:
                        continue
                    box = Bbox([[left, y_lo], [right, y_hi]])
                    if any(
                        box.overlaps(placed_box) for placed_box in others_placed
                    ) or any(
                        path.intersects_bbox(box.padded(half_width))
                        for path, half_width in arrows
                    ):
                        continue
                    nearest_behind = min(behind, default=np.inf)
                    nearest_ahead = min(ahead, default=np.inf)
                    # With the side given, the clearance alone ranks them.
                    away = side != "auto" or (
                        nearest_behind <= nearest_ahead + _LABEL_TIE_PX
                    )
                    clearance = min(nearest_behind, nearest_ahead)
                    if (
                        best_key is None
                        or away > best_key[0]
                        or (
                            away == best_key[0]
                            and clearance > best_key[1] + _LABEL_TIE_PX
                        )
                    ):
                        near_y = y_lo if this_side == "above" else y_hi
                        best_key = (away, clearance)
                        best_place = (left, near_y, this_side)
        if best_place is None:
            return None
        left, near_y, this_side = best_place
        x_data, y_data = ax.transData.inverted().transform((left, near_y))
        # Anchored at its near edge, which the canvas and a PDF then share,
        # though the label's height differs between them.
        label.set_verticalalignment("bottom" if this_side == "above" else "top")
        label.set_transform(ax.transData)
        label.set_position((float(x_data), float(y_data)))
        placed = True
        return label
    finally:
        if not placed:
            label.remove()


# Contour levels.
# Tolerance, in units of `step`, for treating a value as sitting exactly on a
# grid multiple.  Without it, floor and ceil step a whole band
# too far whenever the quotient is an exact integer mathematically but not in
# binary floating point (0.3 / 0.1 is 2.9999999999999996; 0.07 / 0.01 is
# 7.000000000000001), producing the empty end band this function exists to
# avoid.
_GRID_SNAP_TOL = 1e-9


def tight_levels(arrs: ArrayLike | Sequence[ArrayLike], step: float) -> np.ndarray:
    """Contour levels bracketing the data, with no unused band at either end.

    The levels are the multiples of ``step`` from the largest one at or below
    the data minimum through the smallest one at or above the data maximum.
    The two end bands therefore contain plotted values, so a colorbar drawn
    from these levels has no empty box on the low or the high end; a band in
    between is empty wherever no values fall in its range.  A field
    spanning zero gets zero as one of its levels, the grid being anchored
    there, which keeps the levels usable for a diverging colormap centered on
    zero via ``trunc_cmap_about_center``.

    The bracketing is asymmetric about zero whenever the data are: a field
    running -0.054 to +0.029 with ``step=0.01`` yields levels -0.06 to +0.03.

    A data extremum landing exactly on a multiple of ``step`` is treated as
    being on the grid rather than a hair outside it, so it adds no empty band.
    Constant data yield two levels rather than one, since a single level is
    zero bands and no contour routine can use it.

    NaNs, infinities and the masked cells of a masked array are ignored.  A
    field with no finite values at all raises, having nothing to bracket.

    Parameters
    ----------
    arrs : array-like or sequence of array-like
        Field to be contoured, restricted beforehand to the domain that will
        actually be displayed.  Pass a list or tuple of fields to pool them,
        as for multiple panels sharing one colorbar.
    step : float
        Spacing between adjacent levels.

    Returns
    -------
    numpy.ndarray
        The contour levels, in increasing order.

    """
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")
    _, min_val, max_val = _finite_values(arrs)
    start = -_ceil_index(-min_val, step)
    num_bands = max(_ceil_index(max_val, step) - start, 1)
    return cast(np.ndarray, start * step + step * np.arange(num_bands + 1))


def _ceil_index(value: float, step: float) -> int:
    """The index of the smallest multiple of ``step`` at or above ``value``.

    A value within ``_GRID_SNAP_TOL`` times ``step`` of a multiple counts as
    on it.  Negating both value and index gives the largest multiple at or
    below.
    """
    return int(np.ceil(value / step - _GRID_SNAP_TOL))


# Colormaps.
def truncate_cmap(cmap, minval=0.0, maxval=1.0, n_colors=None):
    """Truncate a colormap.

    From https://stackoverflow.com/a/18926541/1706640.

    """
    if n_colors is None:
        try:
            n_colors = len(cmap.colors)
        except AttributeError:
            n_colors = 100
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n_colors)),
    )
    return new_cmap


def trunc_cmap_about_center(
    cmap, min_val=None, max_val=None, central_val=0, arr=None, n_colors=None
):
    """Diverging colormap centered at specified value w/ asymmetric extent.

    I.e. stops at the specified minimum and maximum values while retaining
    the given colormap's built-in spacing.

    """
    if min_val is None:
        min_val = arr.min().values
    if max_val is None:
        max_val = arr.max().values
    min_central_diff = central_val - min_val
    max_central_diff = max_val - central_val
    if min_central_diff > max_central_diff:
        cmap_min = 0.0
        cmap_max = 0.5 * (max_val - min_val) / min_central_diff
    else:
        cmap_min = 1.0 - 0.5 * (max_val - min_val) / max_central_diff
        cmap_max = 1.0
    return truncate_cmap(cmap, minval=cmap_min, maxval=cmap_max, n_colors=n_colors)


def band_colors_about_center(
    cmap: Colormap,
    levels: ArrayLike | Sequence[float],
    central_val: float = 0.0,
    extend: str = "neither",
) -> list[tuple[float, float, float, float]]:
    """Colors for filled contour bands diverging about a central value.

    Discrete-band counterpart to ``trunc_cmap_about_center``, for a filled
    contour plot whose levels are asymmetric about ``central_val``.  Each band
    takes the color at its own center, with the color increment per band set by
    the longer of the two sides, so the shorter side stops short of that end of
    the colormap: a given increment in the field is a given increment in color
    on both sides of the center.

    The contour at ``central_val`` lands exactly on the colormap's midpoint
    color.  Handing the truncated colormap to ``contourf`` instead leaves that
    contour up to half a band off the midpoint, because matplotlib samples a
    discretized colormap at the band edges rather than their centers.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Diverging colormap, whose midpoint color marks ``central_val``.
    levels : array-like
        Contour levels, increasing.  ``pplt.tight_levels`` generates ones that
        bracket the data and include ``central_val`` when the data span it.
    central_val : float
        Field value to place at the colormap's midpoint color.
    extend : {"neither", "min", "max", "both"}
        Which ends of the colorbar carry an arrow tip, as matplotlib's
        ``extend`` takes it.  Each tip takes the color one band further along
        its side of the colormap than that side's outermost band, clipped to
        the colormap's end.  So the longer side's tip is the colormap's end
        color, and the shorter side's tip continues its sequence and stays
        paler than the longer side's colors of the same magnitude.

    Returns
    -------
    list
        One RGBA tuple per band, i.e. ``len(levels) - 1`` of them, with the
        lower tip's color before them and the upper tip's after them, as
        ``extend`` asks.  That is the order ``ListedColormap`` and
        ``BoundaryNorm(levels, len(colors), extend=extend)`` take them in; with
        no tips, pass them to ``contourf`` as ``colors``.

    """
    level_vals = _increasing_levels(levels)
    if extend not in _EXTENDS:
        raise ValueError(f"extend must be one of {_EXTENDS}; got {extend!r}")
    ind_center = int(np.argmin(np.abs(level_vals - central_val)))
    num_below, num_above = ind_center, level_vals.size - 1 - ind_center
    num_long = max(num_below, num_above)
    band_centers = np.arange(-num_below, num_above) + 0.5
    colors_out = [cmap(0.5 + center / (2 * num_long)) for center in band_centers]
    if extend in ("min", "both"):
        position = 0.5 + (-num_below - 0.5) / (2 * num_long)
        colors_out.insert(0, cmap(float(np.clip(position, 0.0, 1.0))))
    if extend in ("max", "both"):
        position = 0.5 + (num_above + 0.5) / (2 * num_long)
        colors_out.append(cmap(float(np.clip(position, 0.0, 1.0))))
    return colors_out


def _increasing_levels(levels: ArrayLike | Sequence[float]) -> np.ndarray:
    """``levels`` as a float array, checked to have two or more increasing values."""
    level_vals = np.asarray(levels, dtype=float)
    if level_vals.size < 2:
        raise ValueError(f"need at least 2 levels; got {level_vals.size}")
    if not np.all(np.diff(level_vals) > 0):
        raise ValueError(f"levels must increase and contain no NaNs; got {level_vals}")
    return level_vals


def sequential_band_colors(
    cmap: Colormap,
    levels: ArrayLike | Sequence[float],
    *,
    extend: str = "neither",
    start: float = 0.1,
) -> list[tuple[float, float, float, float]]:
    """Colors for the filled contour bands of a sequential colormap.

    The sequential counterpart of ``band_colors_about_center``, for a field of
    one sign.  The bands are sampled evenly from ``start`` to the colormap's
    end.  The default ``start`` of 0.1 skips the white end of maps such as
    Crameri's ``davos_r``, whose first color matches that of a masked cell.

    With an upper arrow tip, the last of ``n`` bands is taken at
    ``1 - 0.8 / n``, or a single band at ``start``, and the tip at the
    colormap's end, so the tip lies further along the colormap than every
    band.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Sequential colormap, running from the color of the smallest values to
        that of the largest.
    levels : array-like
        Contour levels, increasing.  ``tight_scale`` with ``kind="sequential"``
        chooses them.
    extend : {"neither", "max"}
        Whether the colorbar carries an upper arrow tip.  A sequential bar from
        ``tight_scale`` starts at or below its smallest value, so it never tips
        at its lower end, and a lower tip raises a ``ValueError``.
    start : float
        Position on ``cmap`` of the lightest band, from 0 to 1.

    Returns
    -------
    list
        One RGBA tuple per band, then the tip's color if ``extend`` is
        ``"max"``, the order ``ListedColormap`` and ``BoundaryNorm(levels,
        len(colors), extend=extend)`` take them in.

    """
    level_vals = _increasing_levels(levels)
    if extend not in ("neither", "max"):
        raise ValueError(
            f"a sequential bar tips only at its upper end; got extend={extend!r}"
        )
    num_bands = level_vals.size - 1
    end = 1.0 - 0.8 / num_bands if extend == "max" else 1.0
    if not 0.0 <= start < end:
        raise ValueError(
            f"start must lie in [0, {end:g}), below the last band; got {start}"
        )
    colors_out = [cmap(float(pos)) for pos in np.linspace(start, end, num_bands)]
    if extend == "max":
        colors_out.append(cmap(1.0))
    return colors_out


# Colorbar scales: Spencer Hill's rules of 2026-09-28 for every colorbar, moved
# here on 2026-10-22 from the lps-enso-grl project's `figure_style.tight_scale`.
BANDS_PER_SIDE = 8
"""Bands on the longer side of zero of a diverging colorbar.

So a bar symmetric about zero has sixteen bands, and an asymmetric one has
eight on its longer side and as many as reach its extreme on the shorter.
"""

SEQUENTIAL_BANDS = 16
"""The most bands a sequential colorbar has, the sixteen of a symmetric
diverging one."""

TIP_QUANTILE = 0.95
"""The quantile of magnitude that a colorbar side's arrow-tip test refers to;
see ``tight_scale``."""

TIP_RATIO = 1.3
"""How far, as a multiple of its ``TIP_QUANTILE`` reference, a side's extreme
may run before that side ends in an arrow tip; see ``tight_scale``."""

MAX_TICK_LABELS = 9
"""The most labels ``colorbar_ticks`` puts on a colorbar by default: every
second boundary of a sixteen-band bar, including both ends and zero."""

# Leading digits a diverging band width may take, times a power of ten.  Each
# is at most 1.5 times the one before, so eight bands of the chosen width end
# near the extent they have to cover.
_STEP_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
# Coarser, because a sequential bar is labeled from its ends rather than from
# zero, and a band width of 0.8 or 3 would label it 0.8, 2.4, 4.0 or 3, 12, 21.
_SEQUENTIAL_MANTISSAS = (1.0, 2.0, 2.5, 5.0)
_EXTENDS = ("neither", "min", "max", "both")


@dataclass(frozen=True, eq=False)
class TightScale:
    """The contour levels ``tight_scale`` chooses for one colorbar.

    Attributes
    ----------
    levels : numpy.ndarray
        Band edges, increasing.  Each is a multiple of the band width rounded
        to six significant figures of it, so that a level such as -2.4 is not
        -2.4000000000000004, unless that rounding would move it off the grid,
        as for a multiple of 1/3.  An end level the data's extreme passes by
        no more than rounding is set to that extreme, as is the lowest level
        of a sequential bar whose smallest value is below zero by rounding, so
        that a contour routine fills those cells and no arrow tip is drawn for
        them.
    extend : str
        Which ends of the colorbar carry an arrow tip, as matplotlib's
        ``extend`` takes it: ``"neither"``, ``"min"``, ``"max"`` or ``"both"``.
    extremes : tuple of float
        The smallest and largest finite values over every field, which an
        arrow tip should print.
    step : float
        The band width.

    """

    levels: np.ndarray
    extend: str
    extremes: tuple[float, float]
    step: float


def _finite_values(
    arrs: ArrayLike | Sequence[ArrayLike],
) -> tuple[list[np.ndarray], float, float]:
    """Each field's finite values, flattened, and the smallest and largest of
    them all.

    The values come one array per field that has any.  Each item of a list or
    tuple is one field, except that a list or tuple of scalars alone is a
    single field.  The masked cells of a masked array are dropped with the
    NaNs and infinities.  No finite value in any field raises a
    ``ValueError``.
    """
    arr_seq: list[Any]
    if isinstance(arrs, list | tuple):
        scalars = all(np.ndim(item) == 0 for item in arrs)
        arr_seq = [np.asarray(arrs, dtype=float)] if scalars else list(arrs)
    else:
        arr_seq = [arrs]
    parts = []
    for arr in arr_seq:
        vals: np.ndarray = np.ma.filled(np.ma.asarray(arr, dtype=float), np.nan).ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size:
            parts.append(finite)
    if not parts:
        raise ValueError("no finite values in `arrs`")
    lo = min(float(vals.min()) for vals in parts)
    hi = max(float(vals.max()) for vals in parts)
    return parts, lo, hi


def _side_reference(parts: list[np.ndarray], sign: int, quantile: float) -> float:
    """The largest, over the fields, of each one's ``quantile`` of magnitude on
    one side of zero.  ``sign`` of 0 takes the values themselves."""
    refs = []
    for vals in parts:
        side = vals if sign == 0 else (vals[vals > 0] if sign > 0 else -vals[vals < 0])
        if side.size:
            refs.append(float(np.quantile(side, quantile)))
    return max(refs)


def _covered(extreme: float, reference: float, ratio: float) -> float:
    """What a side's bands cover: its reference if its extreme runs more than
    ``ratio`` times past it, and otherwise the extreme itself."""
    return reference if extreme > ratio * reference else extreme


def _round_to_step(values: ArrayLike, step: float) -> np.ndarray:
    """Rounded to six significant figures of ``step``, so that -6 times 0.4 is
    -2.4 and not -2.4000000000000004, however far the values lie from zero.

    A value that rounding would move by more than ``_GRID_SNAP_TOL`` times
    ``step`` has no short decimal form, as for a multiple of 1/3, and is left
    as it is, since rounding would move it off the grid.
    """
    vals = np.asarray(values, dtype=float)
    decimals = int(6 - np.floor(np.log10(step)))
    rounded = np.round(vals, decimals)
    return cast(
        np.ndarray,
        np.where(np.abs(rounded - vals) <= _GRID_SNAP_TOL * step, rounded, vals),
    )


def _snap_ends(levels: np.ndarray, lo: float, hi: float, step: float) -> np.ndarray:
    """``levels`` with an end moved onto the data's extreme where the extreme
    passes it by no more than rounding.

    Such an extreme lies on the grid in all but its last bits.  Left outside,
    it would make a spurious arrow tip, or leave the cell unfilled by a
    contour routine; snapped, it fills the end band.
    """
    tol = _GRID_SNAP_TOL * step
    out = levels.copy()
    if out[0] - tol <= lo < out[0]:
        out[0] = lo
    if out[-1] < hi <= out[-1] + tol:
        out[-1] = hi
    return out


def _nice_steps(width: float, mantissas: Sequence[float]) -> list[float]:
    """``mantissas`` times the three powers of ten around ``width``, ascending."""
    exponent = int(np.floor(np.log10(width)))
    return sorted(
        float(f"{m * 10.0**e:.12g}")
        for e in (exponent - 1, exponent, exponent + 1)
        for m in mantissas
    )


def _long_side_step(cover: float, largest: float, num_bands: int) -> float:
    """The band width putting ``num_bands`` bands on the longer side of zero.

    The smallest nice width whose bands reach ``cover``, unless ``largest``,
    the side's extreme, does not pass ``num_bands - 1`` of those widths, so
    that no value would fall in the outermost band.  Then it is the next smaller
    nice width, and the rest of that side runs into an arrow tip.  Reaching
    and passing are judged as ``tight_levels`` judges them, with a value
    within ``_GRID_SNAP_TOL`` of a band edge counted as on it.
    """
    steps = _nice_steps(cover / num_bands, _STEP_MANTISSAS)
    up = min(s for s in steps if _ceil_index(cover, s) <= num_bands)
    if _ceil_index(largest, up) >= num_bands:
        return up
    return max(s for s in steps if s < up)


def _sequential_step(lo: float, hi: float, num_bands: int) -> float:
    """The finest nice sequential band width giving at most ``num_bands``
    bands from ``lo`` to ``hi``, as ``tight_levels`` counts them, looking a
    decade coarser each time none of the widths around the first estimate
    does."""
    width = (hi - lo) / num_bands
    while True:
        for step in _nice_steps(width, _SEQUENTIAL_MANTISSAS):
            if _ceil_index(hi, step) + _ceil_index(-lo, step) <= num_bands:
                return step
        width *= 10.0


def _extend_for(lo: float, hi: float, levels: np.ndarray) -> str:
    """Which ends of ``levels`` the data, from ``lo`` to ``hi``, run past, as
    matplotlib's ``extend``."""
    return {
        (False, False): "neither",
        (True, False): "min",
        (False, True): "max",
        (True, True): "both",
    }[(bool(lo < levels[0]), bool(hi > levels[-1]))]


def tight_scale(
    arrs: ArrayLike | Sequence[ArrayLike],
    *,
    kind: str = "diverging",
    step: float | None = None,
    bands_per_side: int = BANDS_PER_SIDE,
    num_bands: int = SEQUENTIAL_BANDS,
    quantile: float = TIP_QUANTILE,
    tip_ratio: float = TIP_RATIO,
) -> TightScale:
    """Contour levels for a colorbar covering only the values actually plotted.

    Implements Spencer Hill's colorbar rules of 2026-09-28:

    1. A bar spans only the range of values plotted across all the fields it
       serves, so the color at each end is used: by the data's extreme, or,
       where that end carries an arrow tip, by the values past it.  Bands in
       between may go unused, as when one outlier sits far from the rest.  A
       diverging bar is asymmetric about zero wherever the data are, and
       always reaches zero.  The levels are multiples of one band width, as
       ``tight_levels`` gives.
    2. A diverging bar has ``bands_per_side`` bands, eight, on its longer side
       of zero, and on its shorter side as many as reach that side's extreme.
       The longer side is the one whose covered extent, below, is larger.
    3. A side ends in an arrow tip when its extreme runs more than
       ``tip_ratio`` times past its reference, the largest over the fields of
       each field's own ``quantile`` of magnitude on that side; its bands then
       cover the reference instead of the extreme.  Taking the quantile per
       field keeps a field of small values sharing the bar from pulling the
       reference down, as a quantile of all the fields pooled would.
    4. An asymmetric bar never tips on its shorter side, which always reaches
       its own extreme.  With ``band_colors_about_center`` the shorter side's
       colors stop short of the longer side's, so a tip there would draw that
       side's most extreme values paler than smaller magnitudes on the other
       side.  When the shorter side would need as many bands as the longer,
       the bar becomes symmetric, and then either side may tip.

    The band width is a nice number: 1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a
    power of ten.  When the smallest nice width whose eight bands reach the
    longer side's covered extent would leave its outermost band empty, the
    next smaller width is taken, and the rest of that side runs into a tip
    even if its extreme is within ``tip_ratio`` of its reference.  On a field
    whose values reach their extreme in a broad plateau, that tip can contain
    many cells: for 10.4 sin(x) over one period, whose extreme is 1.003 times
    its 95th percentile on each side, the bar spans -8 to 8 and each tip
    contains 22% of all the cells.

    A sequential bar takes up to ``num_bands`` bands from the smallest value
    to its covered top, chosen by the same tip test on the values themselves,
    and only its top can tip.  Since the test compares values with zero, a
    field far from zero, such as a temperature in Kelvin, seldom tips; scale
    its departure from a baseline instead.

    The levels are evenly spaced, which suits a field spanning orders of
    magnitude, such as precipitation, poorly.  Where most cells have values
    near zero, the reference falls among them, so the bands cover only those
    values and every larger one lands in the tip.  For 960 values from 0 to
    0.01 and 40 from 0.5 to 40, the 95th percentile is 0.0099, and the
    sequential bar spans 0 to 0.01 with all 40 larger values in its tip.
    Mask the values near zero first, or use log-spaced levels, which this
    function does not give.

    A diverging bar is centered on zero, so a field of one sign gets bands
    from zero to its extreme.  For another central value, pass the fields
    minus that value and add it back to the levels.

    Parameters
    ----------
    arrs : array-like or sequence of array-like
        The field, or a list or tuple of every field the colorbar serves, each
        restricted beforehand to what its panel draws: its frame and its mask.
        Cells passed from outside the frame widen the bar by values nobody
        sees.  NaNs, infinities and masked cells are ignored, and a list or
        tuple of plain numbers is one field.
    kind : {"diverging", "sequential"}
        A diverging bar suits a field whose plotted values span zero; a
        sequential one, a field of one sign, which must not be negative beyond
        rounding (1e-9 of its largest magnitude).
    step : float, optional
        The band width, positive and finite.  By default it is chosen from
        nice widths to give the band counts above.  A diverging bar with
        ``step`` given has as many bands on its longer side as reach that
        side's covered extent.
    bands_per_side : int
        Bands on the longer side of zero of a diverging bar, an integer of at
        least 1.
    num_bands : int
        The most bands of a sequential bar, an integer of at least 1.
    quantile : float
        The quantile of magnitude each side's arrow-tip test refers to,
        greater than 0 and at most 1.
    tip_ratio : float
        How far past that reference a side's extreme may run before the side
        ends in an arrow tip; positive.

    Returns
    -------
    TightScale
        The levels, the ``extend`` they need, the extremes an arrow tip should
        print, and the band width.  ``band_colors_about_center`` or
        ``sequential_band_colors`` give the band colors, and
        ``colorbar_ticks`` the boundaries to label.

    Raises
    ------
    ValueError
        If ``kind`` is neither ``"diverging"`` nor ``"sequential"``; if a
        parameter lies outside the range given above; if the fields have no
        finite values, or, with no ``step``, only one value or values that
        differ only by rounding; or if a sequential field has a negative
        value beyond rounding.

    """
    if kind not in ("diverging", "sequential"):
        raise ValueError(f"kind must be 'diverging' or 'sequential'; got {kind!r}")
    if step is not None and not 0 < step < np.inf:
        raise ValueError(f"step must be positive and finite; got {step}")
    if (
        not isinstance(bands_per_side, Integral)
        or not isinstance(num_bands, Integral)
        or bands_per_side < 1
        or num_bands < 1
    ):
        raise ValueError(
            f"bands_per_side and num_bands must be integers of at least 1; got "
            f"{bands_per_side} and {num_bands}"
        )
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must lie in (0, 1]; got {quantile}")
    if not tip_ratio > 0:
        raise ValueError(f"tip_ratio must be positive; got {tip_ratio}")
    parts, lo, hi = _finite_values(arrs)
    if step is None and lo == hi:
        raise ValueError("the fields have a single value; pass `step`")
    if kind == "sequential":
        if lo < -1e-9 * max(abs(lo), abs(hi)):
            raise ValueError(
                f"a sequential field must not be negative; its smallest value is "
                f"{lo:g}.  A field spanning zero takes kind='diverging'."
            )
        # A smallest value below zero by no more than rounding counts as zero:
        # the bands start there, and the lowest level is then moved down to
        # take that value in.
        seq_lo = max(lo, 0.0)
        cover_hi = _covered(hi, _side_reference(parts, 0, quantile), tip_ratio)
        if cover_hi <= seq_lo:
            # A reference at the smallest value, as for a field mostly at
            # zero, leaves nothing for the bands to cover short of the extreme.
            cover_hi = hi
        if step is None:
            step = _sequential_step(seq_lo, cover_hi, num_bands)
        levels = _round_to_step(tight_levels(np.array([seq_lo, cover_hi]), step), step)
        levels[0] = min(levels[0], lo)
    else:
        extreme = {+1: max(hi, 0.0), -1: max(-lo, 0.0)}
        cover = {
            sign: (
                _covered(
                    extreme[sign], _side_reference(parts, sign, quantile), tip_ratio
                )
                if extreme[sign] > 0
                else 0.0
            )
            for sign in (+1, -1)
        }
        long_side = +1 if cover[+1] >= cover[-1] else -1
        short_side = -long_side
        if step is None:
            step = _long_side_step(cover[long_side], extreme[long_side], bands_per_side)
            num_long = bands_per_side
        else:
            num_long = max(_ceil_index(cover[long_side], step), 1)
        num_short = min(_ceil_index(extreme[short_side], step), num_long)
        counts = {long_side: num_long, short_side: num_short}
        levels = _round_to_step(step * np.arange(-counts[-1], counts[+1] + 1), step)
    levels = _snap_ends(levels, lo, hi, step)
    if not np.all(np.diff(levels) > 0):
        raise ValueError(
            "the fields' values differ only by rounding, too little for "
            "distinct levels; pass `step`"
        )
    return TightScale(
        levels=levels,
        extend=_extend_for(lo, hi, levels),
        extremes=(lo, hi),
        step=float(step),
    )


def _zero_index(levels: np.ndarray) -> int | None:
    """Index of the level at zero, to within rounding, or None if there is none."""
    scale = float(np.max(np.abs(levels)))
    hits = np.flatnonzero(np.abs(levels) <= 1e-9 * max(scale, 1e-300))
    return int(hits[0]) if hits.size else None


def colorbar_ticks(
    levels: ArrayLike | Sequence[float],
    every: int = 2,
    max_labels: int = MAX_TICK_LABELS,
) -> list[float]:
    """The band boundaries of a colorbar to label.

    Labels every ``every``-th boundary or coarser, with at most ``max_labels``
    labels, in one of three ways.

    1. **Zero inside the bar:** stepping outward from zero, every ``every``-th
       boundary, or the first coarser stride giving at most ``max_labels``.
       Stepping from the ends instead leaves zero unlabeled whenever the two
       sides differ in band count, as most bars from ``tight_scale`` do.  A
       short bar may get only zero, or zero and one end; an ``every`` past
       the number of bands labels zero alone.
    2. **Otherwise,** at a stride dividing the number of bands, so both ends
       are labeled and every label sits the same number of bands from its
       neighbors.  Among the strides giving 3 to ``max_labels`` labels, the
       one nearest ``every`` is taken, coarser on a tie; when zero is an end
       level, a stride giving at least four labels is preferred.
    3. **When no such stride exists,** as when a prime number of bands rules
       out every divisor but 1, at the stride nearest ``every`` giving 3 to
       ``max_labels`` labels, counted from zero if zero is a level and from
       the lowest boundary if not.  One end is then left unlabeled.  If no
       stride gives three labels, the two ends are labeled.

    Each boundary returned is rounded to six significant figures of the
    narrowest band, where that moves it by no more than rounding, and a level
    within rounding of zero is labeled as exactly 0.  So an end level a hair
    off a round value is labeled as that value, and a level with no short
    decimal form, such as 2/3, is labeled where it lies.

    The levels are taken to be evenly spaced, as ``tight_scale`` gives them.
    The test for a level at zero is relative to the largest level: a level
    no further from zero than 1e-9 times the largest magnitude counts as
    zero.  So of log-spaced levels from 0.001 to 1e6, the level 0.001 is
    labeled 0.

    Parameters
    ----------
    levels : array-like
        The band edges, increasing, as ``tight_scale`` gives them.
    every : int
        The finest stride, in bands, between labeled boundaries.
    max_labels : int
        The most labels, at least 3.

    Returns
    -------
    list of float
        The boundaries to pass to ``Colorbar.set_ticks``.

    """
    level_vals = _increasing_levels(levels).copy()
    if every < 1:
        raise ValueError(f"every must be at least 1; got {every}")
    if max_labels < 3:
        raise ValueError(f"max_labels must be at least 3; got {max_labels}")
    num = level_vals.size - 1
    zero = _zero_index(level_vals)
    if zero is not None:
        level_vals[zero] = 0.0
    level_vals = _round_to_step(level_vals, float(np.min(np.diff(level_vals))))
    if zero is not None and 0 < zero < num:
        for stride in range(min(every, num), num + 1):
            picked = level_vals[zero % stride :: stride]
            if picked.size <= max_labels:
                return [float(v) for v in picked]
    divisors = [d for d in range(1, num + 1) if num % d == 0]
    pool = [d for d in divisors if 3 <= num // d + 1 <= max_labels]
    if not pool:
        anchor = 0 if zero is None else zero
        strides = [
            k
            for k in range(1, num + 1)
            if 3 <= len(range(anchor % k, num + 1, k)) <= max_labels
        ]
        if not strides:
            return [float(level_vals[0]), float(level_vals[-1])]
        stride = min(strides, key=lambda k: (abs(k - every), -k))
        return [float(v) for v in level_vals[anchor % stride :: stride]]
    # Zero, if a level here, is an end, which every divisor labels.
    preferred = [d for d in pool if zero is not None and num // d + 1 >= 4]
    stride = min(preferred or pool, key=lambda d: (abs(d - every), -d))
    return [float(v) for v in level_vals[::stride]]


# Plot horizontal, vertical, diagonal lines.
def mark_x0(ax=None, linewidth=0.5, color="0.5", x0=0, **kwargs):
    """Mark the x intercept line on the given axis."""
    ax = _gca_if_ax_none(ax)
    return ax.axvline(x=x0, linewidth=linewidth, color=color, **kwargs)


def mark_y0(ax=None, linewidth=0.5, color="0.5", y0=0, **kwargs):
    """Mark the y intercept on the given axis."""
    ax = _gca_if_ax_none(ax)
    return ax.axhline(y=y0, linewidth=linewidth, color=color, **kwargs)


def mark_one2one(
    ax=None, *line_args, linestyle=":", color="0.7", linewidth=0.8, **line_kwargs
):
    """Mark the identity line, y=x, on the given axis.

    From https://stackoverflow.com/a/28216751/1706640.

    """
    ax = _gca_if_ax_none(ax)
    (identity,) = ax.plot(
        [], [], *line_args, linestyle=linestyle, color=color, **line_kwargs
    )

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect("xlim_changed", callback)
    ax.callbacks.connect("ylim_changed", callback)
    return ax


# Timeseries plots.
def plot_ts_compar(arr1, arr2, dim="year", fig=None, axarr=None, **faceted_kwargs):
    """Compare statistics of two 1-D arrays via various plots."""
    if fig is None and axarr is None:
        fig, axarr = faceted(
            3, 2, width=8, sharex=False, sharey=False, **faceted_kwargs
        )

    def identity(x, *args, **kwargs):
        return x

    color_arr1 = "blue"
    color_arr2 = "orange"

    # Columns: left is raw, right is detrended standardized anomalies
    # Row 1: plot both timeseries,
    funcs = identity, dt_std_anom
    labels = "raw", "detrended std. anoms."
    for n, (ax, func, label) in enumerate(zip(axarr[:2], funcs, labels)):
        arr_plot1 = func(arr1, dim=dim)
        arr_plot2 = func(arr2, dim=dim)
        arr_plot1.plot(ax=ax, color=color_arr1)
        arr_plot2.plot(ax=ax, color=color_arr2)
        extra_text = f"{label}, r={float(xr.corr(arr_plot1, arr_plot2)):.2f}"
        panel_label(n, ax=ax, extra_text=extra_text)

        # Overlay mean and trend in first panel.
        if n == 0:
            mean1_plot = xr.ones_like(arr1) * arr1.mean(dim)
            mean2_plot = xr.ones_like(arr2) * arr2.mean(dim)
            mean1_plot.plot(ax=axarr[0], linestyle=":", color=color_arr1)
            mean2_plot.plot(ax=axarr[0], linestyle=":", color=color_arr2)
            trend(arr1, dim=dim).plot(ax=axarr[0], linestyle="--", color=color_arr1)
            trend(arr2, dim=dim).plot(ax=axarr[0], linestyle="--", color=color_arr2)

    # Row 2: histograms
    for n, (ax, func, label) in enumerate(zip(axarr[2:4], funcs, labels)):
        arr_plot1 = func(arr1, dim=dim)
        arr_plot2 = func(arr2, dim=dim)
        arr_plot1.plot.hist(ax=ax, color=color_arr1)
        arr_plot2.plot.hist(ax=ax, color=color_arr2, alpha=0.5)
        stdev1 = float(arr_plot1.std(dim))
        stdev2 = float(arr_plot2.std(dim))
        extra_text = (
            f"{label}, "
            + r"$\sigma_1$="
            + f"{stdev1:.2f}, "
            + r"$\sigma_2$="
            + f"{stdev2:.2f}"
        )
        panel_label(n + 2, ax=ax, extra_text=extra_text)

    # Row 3: scatterplots and linear regression
    funcs = identity, detrend
    labels = "raw", "detrended"
    for n, (ax, func, label) in enumerate(zip(axarr[4:], funcs, labels)):
        arr_plot1 = func(arr1, dim=dim)
        arr_plot2 = func(arr2, dim=dim)

        # Overlay ordinary least squares regression.
        regress = lin_regress(arr_plot1, arr_plot2, dim)
        rsquare = float(regress.sel(parameter="r_value") ** 2)
        slope = float(regress.sel(parameter="slope"))
        intercept = float(regress.sel(parameter="intercept"))
        ax.scatter(*xr.align(arr_plot1, arr_plot2), s=8)
        ax.axline(
            (min(arr_plot1), min(arr_plot1) * slope + intercept),
            slope=slope,
            color="0.5",
        )
        extra_text = f"{label}, slope={slope:.2f}, " + r"r$^2$=" + f"{rsquare:.2f}"
        panel_label(n + 4, ax=ax, extra_text=extra_text)

    [ax.set(title="", xlabel="", ylabel="") for ax in axarr]
    return axarr


# Heatmaps.
def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    do_cbar=False,
    cbar_kw=None,
    cbarlabel="",
    top_ticks=False,
    annotate=True,
    annotate_kw=None,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adapted from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    """
    if cbar_kw is None:
        cbar_kw = {}
    if annotate_kw is None:
        annotate_kw = {}
    ax = _gca_if_ax_none(ax)

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if do_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    if top_ticks:
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        annotate_heatmap(im, **annotate_kw)

    return im, cbar


def _corrs_txt_format(x, pos):
    return f"{x:.2f}".replace("0.", ".").replace("1.00", "")


def annotate_heatmap(
    im,
    data=None,
    valfmt=None,
    textcolors=("black", "white"),
    threshold=None,
    include_diag=False,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Adapted from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.

    """
    if not isinstance(data, list | np.ndarray):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if valfmt is None:
        valfmt = ticker.FuncFormatter(_corrs_txt_format)
    elif isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if include_diag == "all":
                do_annotate = True
            elif include_diag:
                do_annotate = j <= i
            else:
                do_annotate = j < i
            if do_annotate:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts


# Seasonal plots
def plot_seas_points(arr, ax=None, seas_ordered=None, **kwargs):
    """Plot seasonal values in calendar order on one set of axes."""
    ax = _gca_if_ax_none(ax)
    if seas_ordered is None:
        seas_ordered = arr["season"].values
    xrange = range(4)
    linestyle = kwargs.pop("linestyle", "none")
    marker = kwargs.pop("marker", ".")
    ax.plot(
        xrange,
        arr.sel(season=seas_ordered),
        linestyle=linestyle,
        marker=marker,
        **kwargs,
    )
    ax.set_xticks(xrange)
    ax.set_xticklabels(seas_ordered)


def plot_seas_ann_points(arr_seas, arr_ann, ax=None, seas_ordered=None, **kwargs):
    """Plot seasonal and annual-mean values on one set of axes."""
    ax = _gca_if_ax_none(ax)
    if seas_ordered is None:
        seas_ordered = ["MAM", "JJA", "SON", "DJF"]
    xticklabels = seas_ordered + ["annual"]
    xrange = range(5)
    linestyle = kwargs.pop("linestyle", "none")
    marker = kwargs.pop("marker", ".")
    color = kwargs.pop("color", "0.3")
    ax.plot(
        xrange[:-1],
        arr_seas.sel(season=seas_ordered),
        linestyle=linestyle,
        marker=marker,
        color=color,
        **kwargs,
    )
    ax.plot(
        xrange[-1],
        arr_ann,
        linestyle=linestyle,
        marker=marker,
        color=color,
        **kwargs,
    )
    mark_x0(ax, x0=3.5)
    ax.set_xticks(xrange)
    ax.set_xticklabels(xticklabels)


def fig_ann_seas(**fig_kwargs):
    """5-panel figure of annual at top and 4 seasonal values below."""
    fig = plt.figure(**fig_kwargs)
    gs = gridspec.GridSpec(3, 4)

    ax1 = plt.subplot(gs[0, 1:3])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1, 2:])
    ax4 = plt.subplot(gs[2, :2])
    ax5 = plt.subplot(gs[2, 2:])
    axarr = [ax1, ax2, ax3, ax4, ax5]
    return fig, axarr


def ann_seas_labels(
    axarr,
    xparams_inner=None,
    xparams_outer=None,
    yparams_inner=None,
    yparams_outer=None,
):
    """x and y labels for 5-panel annual + seasonal plots."""
    do_x_outer = [True, False, False, True, True]
    do_x_inner = [False, True, True, False, False]
    do_y_outer = [True, True, False, True, False]
    do_y_inner = [False, False, True, False, True]

    if xparams_outer is not None:
        for ax, do_this_ax in zip(axarr, do_x_outer):
            if do_this_ax:
                ax.set(**xparams_outer)
    if xparams_inner is not None:
        for ax, do_this_ax in zip(axarr, do_x_inner):
            if do_this_ax:
                ax.set(**xparams_inner)
    if yparams_outer is not None:
        for ax, do_this_ax in zip(axarr, do_y_outer):
            if do_this_ax:
                ax.set(**yparams_outer)
    if yparams_inner is not None:
        for ax, do_this_ax in zip(axarr, do_y_inner):
            if do_this_ax:
                ax.set(**yparams_inner)


# Writing figures to disk.
def nb_savefig(name, fig=None, fig_dir="../figs", **kwargs):
    """Save a figure from a notebook into the desired figures directory."""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(fig_dir, name), **kwargs)


if __name__ == "__main__":
    pass
