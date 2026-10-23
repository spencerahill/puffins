#! /usr/bin/env python
"""Helper functions for creating plots."""

import os.path
from collections import namedtuple
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import xarray as xr
from faceted import faceted as fac_faceted
from faceted import faceted_ax as fac_ax
from matplotlib import colors, gridspec, ticker
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.text import Text
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
# box it checks.
_LABEL_KWARGS_SET_HERE = (
    "ha",
    "va",
    "horizontalalignment",
    "verticalalignment",
    "transform",
    "rotation_mode",
    "bbox",
)
# Display pixels by which one candidate must beat another to be preferred, so
# that leads equal but for rounding go to the leftmost.
_LABEL_TIE_PX = 1e-6


def _drawn_range(
    px: np.ndarray, heights: np.ndarray, left: float, right: float
) -> tuple[float, float] | None:
    """Smallest and largest of a curve as drawn, over display x from ``left``
    to ``right``.

    ``px`` is sorted.  The curve's finite points in the span count, and so do
    its values interpolated at the span's two ends, but only along a segment
    whose two points are both finite: a NaN gap, and the stretch past either
    end of the curve, are drawn blank and count as no curve.  None when no
    drawn part of the curve lies in the span.
    """
    i_lo = int(np.searchsorted(px, left, side="left"))
    i_hi = int(np.searchsorted(px, right, side="right"))
    inside = heights[i_lo:i_hi]
    vals = [float(v) for v in inside[np.isfinite(inside)]]
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

    Each candidate puts the label's left edge at one of the labeled curve's
    points, on one side of it, with its near edge ``gap`` points beyond the
    labeled curve's extreme over the label's width plus ``gap`` either side.
    So the labeled curve is at least ``gap`` from the label, and at most
    ``gap`` times the square root of 2, at a corner.  Every other curve must
    be farther than that: over the label's width plus that distance either
    side, it must lie wholly behind the labeled curve's extreme, or wholly
    beyond the label's far edge, by more than that distance from the label.
    So the label is clear of every curve and nearer the labeled one.  Of the
    candidates lying wholly inside the axes and overlapping no text in
    ``avoid``, those on the side away from the nearest other curve come
    first, and among them the one where the labeled curve's extreme stands
    farthest from the nearest other curve is taken, the leftmost on a tie,
    and above on a tie at one point.  A candidate on the side toward the
    nearest curve is taken only when no candidate on the side away from it
    fits.

    A curve counts only where it is drawn: along segments between two finite
    points, so neither across a NaN gap nor past either end of the curve.
    Curves are taken as straight segments through their points, centered on
    the line: markers, a stepped drawstyle and the line's own width are not
    counted, so pass a larger ``gap`` for them.

    Distances are measured in display coordinates after a draw, so they hold
    on the rendered figure whatever the axis scales.  The label is anchored in
    data coordinates while its size and its gap are in points, so draw every
    curve and fix the axis limits, the figure size and the layout first: a
    later change to any of them can bring the label nearer a curve or push it
    outside the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes holding the curves.
    x : array-like
        The curves' shared x values, one-dimensional, as plotted; dates are
        converted by the x axis's own units.
    ys : sequence of array-like
        Every curve on the axes that the label must keep clear of, each with
        one value per ``x``; NaN marks a missing point.
    which : int
        Index in ``ys`` of the curve to label.
    text : str
        The label.
    side : {"auto", "above", "below"}
        Which side of its curve the label goes on, as seen on the screen.
        ``"auto"`` tries both, preferring the side away from the nearest
        other curve as the rules ask.
    gap : float
        Distance, in points, from the labeled curve's line to the label's
        near edge; positive.
    avoid : sequence of matplotlib.text.Text or None
        Labels already placed, which this one must not overlap.  None
        entries, as from earlier calls that placed nothing, are skipped.
    **text_kwargs
        Passed to ``Axes.text``, such as ``color`` and ``fontsize``.  The
        alignment, the transform, the rotation mode and a bounding-box patch
        are set here or would move the drawn text off the box checked, and
        raise a ``ValueError`` if passed.

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
    curves = [np.asarray(y, dtype=float) for y in ys]
    x_vals = np.asarray(ax.xaxis.convert_units(np.asarray(x)), dtype=float)
    if x_vals.ndim != 1 or any(c.shape != x_vals.shape for c in curves):
        raise ValueError("x must be one-dimensional and each curve must match it")
    if not 0 <= which < len(curves):
        raise ValueError(f"which must index ys, of length {len(curves)}; got {which}")
    others_placed = [t for t in avoid if t is not None]
    fig = ax.figure
    fig.canvas.draw()
    # Measured at the axes' center, a point every axis scale can place.
    label = ax.text(
        0.5, 0.5, text, transform=ax.transAxes, ha="left", va="bottom", **text_kwargs
    )
    placed = False
    try:
        extent = label.get_window_extent()
        width, height = extent.width, extent.height
        gap_px = gap * fig.dpi / 72.0
        # The farthest the labeled curve can be from the label, at a corner.
        reach = np.sqrt(2.0) * gap_px
        # Display x from x alone: a NaN y makes the transform's x NaN as well,
        # which would drop the other curves' points where the labeled one is
        # missing.
        y_ref = np.full_like(x_vals, float(np.mean(ax.get_ylim())))
        px = ax.transData.transform(np.column_stack([x_vals, y_ref]))[:, 0]
        order = np.argsort(px)
        px = px[order]
        py = [
            ax.transData.transform(np.column_stack([x_vals, c]))[order, 1]
            for c in curves
        ]
        frame = ax.bbox
        best_key: tuple[bool, float] | None = None
        best_place: tuple[float, float] | None = None
        for left in px[np.isfinite(py[which])]:
            right = left + width
            if left < frame.x0 or right > frame.x1:
                continue
            own_span = (left - gap_px, right + gap_px)
            other_span = (left - reach, right + reach)
            for this_side in ("above", "below") if side == "auto" else (side,):
                sign = 1.0 if this_side == "above" else -1.0
                # Never None: the candidate's own point lies in its span.
                own = cast(
                    tuple[float, float],
                    _drawn_range(px, sign * py[which], *own_span),
                )
                near_edge = own[1] + gap_px
                far_edge = near_edge + height
                behind, ahead = [], []
                for k, curve_py in enumerate(py):
                    if k == which:
                        continue
                    drawn = _drawn_range(px, sign * curve_py, *other_span)
                    if drawn is None:
                        continue
                    if drawn[1] < near_edge - reach:
                        behind.append(own[1] - drawn[1])
                    elif drawn[0] > far_edge + reach:
                        ahead.append(drawn[0] - own[1])
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
                    if any(box.overlaps(t.get_window_extent()) for t in others_placed):
                        continue
                    nearest_behind = min(behind, default=np.inf)
                    nearest_ahead = min(ahead, default=np.inf)
                    away = nearest_behind <= nearest_ahead
                    clearance = min(nearest_behind, nearest_ahead)
                    if (
                        best_key is None
                        or away > best_key[0]
                        or (
                            away == best_key[0]
                            and clearance > best_key[1] + _LABEL_TIE_PX
                        )
                    ):
                        best_key, best_place = (away, clearance), (left, y_lo)
        if best_place is None:
            return None
        x_data, y_data = ax.transData.inverted().transform(best_place)
        label.set_transform(ax.transData)
        label.set_position((float(x_data), float(y_data)))
        placed = True
        return label
    finally:
        if not placed:
            label.remove()


# Contour levels.
# Tolerance, in units of `step`, for treating a data extremum as sitting
# exactly on a grid multiple.  Without it, floor and ceil step a whole band
# too far whenever the quotient is an exact integer mathematically but not in
# binary floating point (0.3 / 0.1 is 2.9999999999999996; 0.07 / 0.01 is
# 7.000000000000001), producing the empty end band this function exists to
# avoid.
_GRID_SNAP_TOL = 1e-9


def tight_levels(arrs: ArrayLike | Sequence[ArrayLike], step: float) -> np.ndarray:
    """Contour levels bracketing the data, with no unused band at either end.

    The levels are the multiples of ``step`` from the largest one at or below
    the data minimum through the smallest one at or above the data maximum.
    Every resulting band therefore contains plotted values, so a colorbar drawn
    from these levels has no empty box on the low or the high end.  A field
    spanning zero gets zero as one of its levels, the grid being anchored
    there, which keeps the levels usable for a diverging colormap centered on
    zero via ``trunc_cmap_about_center``.

    The bracketing is asymmetric about zero whenever the data are: a field
    running -0.054 to +0.029 with ``step=0.01`` yields levels -0.06 to +0.03.

    A data extremum landing exactly on a multiple of ``step`` is treated as
    being on the grid rather than a hair outside it, so it adds no empty band.
    Constant data yield two levels rather than one, since a single level is
    zero bands and no contour routine can use it.

    NaNs and infinities are ignored.  A field with no finite values at all
    raises, having nothing to bracket.

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
    arr_seq = arrs if isinstance(arrs, list | tuple) else [arrs]
    mins, maxs = [], []
    for arr in arr_seq:
        vals = np.asarray(arr, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            mins.append(float(finite.min()))
            maxs.append(float(finite.max()))
    if not mins:
        raise ValueError("no finite values in `arrs`; nothing to bracket")
    min_val, max_val = min(mins), max(maxs)
    start = np.floor(min_val / step + _GRID_SNAP_TOL) * step
    stop = np.ceil(max_val / step - _GRID_SNAP_TOL) * step
    num_bands = max(int(np.round((stop - start) / step)), 1)
    return cast(np.ndarray, start + step * np.arange(num_bands + 1))


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

    Returns
    -------
    list
        One RGBA tuple per band, i.e. ``len(levels) - 1`` of them, to pass to
        ``contourf`` as ``colors``.

    """
    level_vals = np.asarray(levels, dtype=float)
    if level_vals.size < 2:
        raise ValueError(f"need at least 2 levels; got {level_vals.size}")
    if not np.all(np.diff(level_vals) > 0):
        raise ValueError(
            "levels must increase and contain no NaNs; otherwise the band "
            f"nearest `central_val` is not well defined.  Got {level_vals}"
        )
    ind_center = int(np.argmin(np.abs(level_vals - central_val)))
    num_below, num_above = ind_center, level_vals.size - 1 - ind_center
    num_long = max(num_below, num_above)
    band_centers = np.arange(-num_below, num_above) + 0.5
    return [cmap(0.5 + center / (2 * num_long)) for center in band_centers]


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
