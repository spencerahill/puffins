#! /usr/bin/env python
"""Helper functions for creating plots."""

from collections import namedtuple
import os.path

from faceted import faceted as fac_faceted
from faceted import faceted_ax as fac_ax
from matplotlib import pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import xarray as xr

from .names import LAT_STR
from .nb_utils import sindeg
from .stats import detrend, dt_std_anom, lin_regress, trend

_DEGR = r'$^\circ$'
_DEGR_S = _DEGR + 'S'
_DEGR_N = _DEGR + 'N'
GRAY = "0.4"

PlotArr = namedtuple('PlotArr', ['func', 'label', 'plot_kwargs'])

plt_rc_params_custom = {
    "axes.edgecolor": GRAY,  # Make axis spines gray.
    "axes.labelcolor": GRAY,  # Make axis labels gray.
    "axes.spines.top": False,  # Turn off top spine in plots.
    "axes.spines.right": False,  # Turn off right spine in plots.
    "figure.dpi": 100,  # Make inline figures larger in Jupyter notebooks.
    "font.family": "Helvetica",  # Use Helvetica font.
    "legend.frameon": False,  # Turn off box around legend.
    "legend.handlelength": 1.,  # Make legend symbols smaller.
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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if displace:
        ax.spines['left'].set_position(('outward', 20))
        ax.spines['bottom'].set_position(('outward', 20))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def sinlat_xaxis(ax=None, start_lat=-90, end_lat=90, do_ticklabels=False,
                 degr_symbols=False):
    """Make the x-axis be in sin of latitude."""
    ax = _gca_if_ax_none(ax)
    ax.set_xlim([sindeg(start_lat), sindeg(end_lat)])
    if start_lat == 0 and end_lat == 90:
        ax.set_xticks(sindeg([0, 30, 60, 90]))
        ax.set_xticks(sindeg([10, 20, 40, 50, 70, 80]), minor=True)
        if do_ticklabels:
            if degr_symbols:
                ax.set_xticklabels(['EQ', r'30$^\circ$' r'60$^\circ$',
                                    r'90$^\circ$'])
            else:
                ax.set_xticklabels(['EQ', r'30N' r'60N', r'90N'])
    elif start_lat == -90 and end_lat == 90:
        ax.set_xticks(sindeg([-90, -60, -30, 0, 30, 60, 90]))
        minorticks = [-80, -70, -50, -40, -20, -10,
                      10, 20, 40, 50, 70, 80]
        ax.set_xticks(sindeg(minorticks), minor=True)
        if do_ticklabels:
            if degr_symbols:
                ax.set_xticklabels(['90' + _DEGR_S, " ", '30' + _DEGR_S, 'EQ',
                                    '30' + _DEGR_N, " ", '90' + _DEGR_N])
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
            ticklabels = ['EQ', '30' + _DEGR, '60' + _DEGR, '90' + _DEGR]
        else:
            ticklabels = ['EQ', '30N', '60N', '90N']
    elif start_lat == -90 and end_lat == 90:
        ticks = [-90, -60, -30, 0, 30, 60, 90]
        minor_ticks = [-80, -70, -50, -40, -20, -10,
                       10, 20, 40, 50, 70, 80]
        if degr_symbol:
            ticklabels = [f"90{_DEGR_S}", f"60{_DEGR_S}", f"30{_DEGR_S}",
                          "EQ", f"30{_DEGR_N}", f"60{_DEGR_N}", f"90{_DEGR_N}"]
        else:
            ticklabels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]
    elif start_lat == -30 and end_lat == 30:
        ticks = [-30, -20, -10, 0, 10, 20, 30]
        minor_ticks = [-25, -15, -5, 5, 15, 25]
        if degr_symbol:
            ticklabels = [f"30{_DEGR_S}", f"20{_DEGR_S}", f"10{_DEGR_S}",
                          "EQ", f"10{_DEGR_N}", f"20{_DEGR_N}", f"30{_DEGR_N}"]
        else:
            ticklabels = ["30S", "20S", "10S", "EQ", "10N", "20N", "30N"]

    elif start_lat == -45 and end_lat == 45:
        ticks = [-45, -30, -15, 0, 15, 30, 45]
        minor_ticks = [-40, -35, -25, -20, -10, -5, 5, 10, 20, 25, 35, 40]
        if degr_symbol:
            ticklabels = [f"45{_DEGR_S}", f"30{_DEGR_S}", f"15{_DEGR_S}",
                          "EQ", f"15{_DEGR_N}", f"30{_DEGR_N}", f"45{_DEGR_N}"]
        else:
            ticklabels = ["45S", "30S", "15S", "EQ", "15N", "30N", "45N"]
    elif start_lat == -60 and end_lat == 60:
        ticks = [-60, -30, 0, 30, 60]
        minor_ticks = [-50, -40, -20, -10, 10, 20, 40, 50]
        if degr_symbol:
            ticklabels = [f"60{_DEGR_S}", f"30{_DEGR_S}", "EQ",
                          f"30{_DEGR_N}", f"60{_DEGR_N}"]
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
            ticklabels = ['EQ', '30' + _DEGR, '60' + _DEGR, '90' + _DEGR]
        else:
            ticklabels = ['EQ', '30N', '60N', '90N']
    elif start_lat == -90 and end_lat == 90:
        ticks = [-90, -60, -30, 0, 30, 60, 90]
        minor_ticks = [-80, -70, -50, -40, -20, -10,
                       10, 20, 40, 50, 70, 80]
        if degr_symbol:
            ticklabels = [f"90{_DEGR_S}", f"60{_DEGR_S}", f"30{_DEGR_S}",
                          "EQ", f"30{_DEGR_N}", f"60{_DEGR_N}", f"90{_DEGR_N}"]
        else:
            ticklabels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]
    elif start_lat == -30 and end_lat == 30:
        ticks = [-30, -20, -10, 0, 10, 20, 30]
        minor_ticks = [-25, -15, -5, 5, 15, 25]
        if degr_symbol:
            ticklabels = [f"30{_DEGR_S}", f"20{_DEGR_S}", f"10{_DEGR_S}",
                          "EQ", f"10{_DEGR_N}", f"20{_DEGR_N}", f"30{_DEGR_N}"]
        else:
            ticklabels = ["30S", "20S", "10S", "EQ", "10N", "20N", "30N"]

    elif start_lat == -45 and end_lat == 45:
        ticks = [-45, -30, -15, 0, 15, 30, 45]
        minor_ticks = [-40, -35, -25, -20, -10, -5, 5, 10, 20, 25, 35, 40]
        if degr_symbol:
            ticklabels = [f"45{_DEGR_S}", f"30{_DEGR_S}", f"15{_DEGR_S}",
                          "EQ", f"15{_DEGR_N}", f"30{_DEGR_N}", f"45{_DEGR_N}"]
        else:
            ticklabels = ["45S", "30S", "15S", "EQ", "15N", "30N", "45N"]
    elif start_lat == -60 and end_lat == 60:
        ticks = [-60, -30, 0, 30, 60]
        minor_ticks = [-50, -40, -20, -10, 10, 20, 40, 50]
        if degr_symbol:
            ticklabels = [f"60{_DEGR_S}", f"30{_DEGR_S}", "EQ",
                          f"30{_DEGR_N}", f"60{_DEGR_N}"]
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


def ann_cyc_xaxis(ax=None, extra_space=False):
    ax = _gca_if_ax_none(ax)
    if extra_space:
        ax.set_xlim(0.8, 12.2)
    else:
        ax.set_xlim(1, 12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels("JFMAMJJASOND")
    ax.set_xlabel("")


def faceted(*args, width=4, aspect=0.618, **kwargs):
    """Wrapper to faceted.faceted w/ a default aspect ratio."""
    return fac_faceted(*args, width=width, aspect=aspect, **kwargs)


def faceted_ax(*args, width=4, aspect=0.618, **kwargs):
    """Wrapper to faceted.faceted_ax w/ a default aspect ratio."""
    return fac_ax(*args, width=width, aspect=aspect, **kwargs)


def plot_lat_1d(arr, start_lat=-90, end_lat=90, sinlat=False,
                ax=None, lat_str=LAT_STR, ax_labels=False, **plot_kwargs):
    """Plot of the given array as a function of latitude."""
    ax = _gca_if_ax_none(ax)
    arr_plot = arr.where((arr[lat_str] > start_lat) &
                         (arr[lat_str] < end_lat))
    if sinlat:
        lat = sindeg(arr_plot[lat_str])
        sinlat_xaxis(ax, start_lat=start_lat, end_lat=end_lat)
    else:
        lat = arr_plot[lat_str]
        lat_xaxis(ax, start_lat=start_lat, end_lat=end_lat)

    handle = ax.plot(lat, arr_plot, **plot_kwargs)[0]

    _left_bottom_spines_only(ax, displace=False)

    if ax_labels:
        ax.set_xlabel(r'Latitude [$^\circ$]')
        if arr.name:
            ax.set_ylabel(arr.name)

    return handle


def _plot_cutoff_ends(lats, arr, ax=None, **kwargs):
    """Avoid finite-differencing artifacts at endpoints."""
    ax = _gca_if_ax_none(ax)
    ax.plot(lats[2:-2], arr[2:-2], **kwargs)


def panel_label(panel_num=None, ax=None, extra_text=None, x=0.01, y=0.88,
                **text_kwargs):
    ax = _gca_if_ax_none(ax)
    if panel_num is None:
        for n, ax_ in enumerate(ax):
            panel_label(n, ax=ax_, x=x, y=y, extra_text=extra_text,
                        **text_kwargs)
        return
    letters = 'abcdefghijklmnopqrstuvwxyz'
    label = '({})'.format(letters[panel_num])
    if extra_text is not None:
        label += ' {}'.format(extra_text)
    ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)


def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap.

    From https://stackoverflow.com/a/18926541/1706640.

    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def mark_x0(ax=None, linewidth=0.5, color='0.5', x0=0, **kwargs):
    """Mark the x intercept line on the given axis."""
    ax = _gca_if_ax_none(ax)
    return ax.axvline(x=x0, linewidth=linewidth, color=color, **kwargs)


def mark_y0(ax=None, linewidth=0.5, color='0.5', y0=0, **kwargs):
    """Mark the y intercept on the given axis."""
    ax = _gca_if_ax_none(ax)
    return ax.axhline(y=y0, linewidth=linewidth, color=color, **kwargs)


def mark_one2one(ax=None, *line_args, linestyle=':', color='0.7',
                 linewidth=0.8, **line_kwargs):
    """Mark the identity line, y=x, on the given axis.

    From https://stackoverflow.com/a/28216751/1706640.

    """
    ax = _gca_if_ax_none(ax)
    identity, = ax.plot([], [], *line_args, linestyle=linestyle,
                        color=color, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)
    return ax


def plot_ts_compar(arr1, arr2, dim="year", fig=None, axarr=None,
                   **faceted_kwargs):
    """Compare statistics of two 1-D arrays via various plots."""
    if fig is None and axarr is None:
        fig, axarr = faceted(3, 2, width=8, sharex=False, sharey=False,
                             **faceted_kwargs)

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
            trend(arr1, dim=dim).plot(ax=axarr[0], linestyle="--",
                                      color=color_arr1)
            trend(arr2, dim=dim).plot(ax=axarr[0], linestyle="--",
                                      color=color_arr2)

    # Row 2: histograms
    for n, (ax, func, label) in enumerate(zip(axarr[2:4], funcs, labels)):
        arr_plot1 = func(arr1, dim=dim)
        arr_plot2 = func(arr2, dim=dim)
        arr_plot1.plot.hist(ax=ax, color=color_arr1)
        arr_plot2.plot.hist(ax=ax, color=color_arr2, alpha=0.5)
        stdev1 = float(arr_plot1.std(dim))
        stdev2 = float(arr_plot2.std(dim))
        extra_text = (f"{label}, " + r"$\sigma_1$=" +
                      f"{stdev1:.2f}, " + r"$\sigma_2$=" + f"{stdev2:.2f}")
        panel_label(n + 2, ax=ax, extra_text=extra_text)

    # Row 3: scatterplots and linear regression
    funcs = identity, detrend
    labels = "raw", "detrended"
    for n, (ax, func, label) in enumerate(zip(axarr[4:], funcs, labels)):
        arr_plot1 = func(arr1, dim=dim)
        arr_plot2 = func(arr2, dim=dim)

        # Overlay ordinary least squares regression.
        regress = lin_regress(arr_plot1, arr_plot2, dim)
        rsquare = float(regress.sel(parameter="r_value")**2)
        slope = float(regress.sel(parameter="slope"))
        intercept = float(regress.sel(parameter="intercept"))
        ax.scatter(*xr.align(arr_plot1, arr_plot2), s=8)
        ax.axline((min(arr_plot1), min(arr_plot1) * slope + intercept),
                  slope=slope, color="0.5")
        extra_text = (f"{label}, slope={slope:.2f}, " + r"r$^2$=" +
                      f"{rsquare:.2f}")
        panel_label(n + 4, ax=ax, extra_text=extra_text)

    [ax.set(title="", xlabel="", ylabel="") for ax in axarr]
    return axarr


def heatmap(data, row_labels, col_labels, ax=None, do_cbar=False, cbar_kw={},
            cbarlabel="", top_ticks=False, annotate=True, annotate_kw={},
            **kwargs):
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
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        annotate_heatmap(im, **annotate_kw)

    return im, cbar


def _corrs_txt_format(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")


def annotate_heatmap(im, data=None, valfmt=None, textcolors=("black", "white"),
                     threshold=None, include_diag=False, **textkw):
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
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

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
        **kwargs
    )
    ax.set_xticks(xrange)
    ax.set_xticklabels(seas_ordered)


def plot_seas_ann_points(arr_seas, arr_ann, ax=None, seas_ordered=None,
                         **kwargs):
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


def nb_savefig(name, fig=None, fig_dir="../figs", **kwargs):
    """Save a figure from a notebook into the desired figures directory."""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(fig_dir, name), **kwargs)


if __name__ == '__main__':
    pass
