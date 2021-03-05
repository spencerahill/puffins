#! /usr/bin/env python
"""Helper functions for creating plots."""

from collections import namedtuple
import os

from faceted import faceted
from matplotlib import pyplot as plt
import numpy as np

from .names import LAT_STR
from .nb_utils import sindeg

_DEGR = r'$^\circ$'
_DEGR_S = _DEGR + 'S'
_DEGR_N = _DEGR + 'N'


PlotArr = namedtuple('PlotArr', ['func', 'label', 'plot_kwargs'])


plt_rc_params_custom = {
    "axes.edgecolor": "0.4",  # Make axis spines gray.
    "axes.spines.top": False,  # Turn off top spine in plots.
    "axes.spines.right": False,  # Turn off right spine in plots.
    "figure.dpi": 100,  # Make inline figures larger in Jupyter notebooks.
    "font.family": "Helvetica",  # Use Helvetica font.
    "legend.frameon": False,  # Turn off box around legend.
    "mathtext.fontset": "cm",  # Use serifed font in equations.
    "pdf.fonttype": 42,  # Bug workaround: https://stackoverflow.com/a/60384073
    "xtick.color": "0.4",  # Make xticks gray.
    "ytick.color": "0.4",  # Make yticks gray.
}


def default_gca(func):
    """If no axes object is given, use gca() to find the active one."""
    def func_default_gca(*args, **kwargs):
        ax = kwargs.get('ax', None)
        if ax is None:
            kwargs['ax'] = plt.gca()
        return func(*args, **kwargs)
    return func_default_gca


def _left_bottom_spines_only(ax, displace=False):
    """Don't plot top or right border."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if displace:
        ax.spines['left'].set_position(('outward', 20))
        ax.spines['bottom'].set_position(('outward', 20))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def sinlat_xaxis(ax, start_lat=-90, end_lat=90):
    """Make the x-axis be in sin of latitude."""
    ax.set_xlim([sindeg(start_lat), sindeg(end_lat)])
    if start_lat == 0 and end_lat == 90:
        ax.set_xticks(sindeg([0, 30, 60, 90]))
        ax.set_xticks(sindeg([10, 20, 40, 50, 70, 80]), minor=True)
        ax.set_xticklabels(['EQ', r'30$^\circ$' r'60$^\circ$', r'90$^\circ$'])
    elif start_lat == -90 and end_lat == 90:
        ax.set_xticks(sindeg([-90, -60, -30, 0, 30, 60, 90]))
        minorticks = [-80, -70, -50, -40, -20, -10,
                      10, 20, 40, 50, 70, 80]
        ax.set_xticks(sindeg(minorticks), minor=True)
        ax.set_xticklabels(['90' + _DEGR_S, " ", '30' + _DEGR_S, 'EQ',
                            '30' + _DEGR_N, " ", '90' + _DEGR_N])


def lat_xaxis(ax, start_lat=-90, end_lat=90, degr_symbol=False):
    """Make the x-axis be latitude."""
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
    else:
        ticks = np.arange(start_lat, end_lat + 1, 10)
    ax.set_xticks(ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(ticklabels)
    ax.set_xlabel(" ")


def lat_yaxis(ax, start_lat=-90, end_lat=90):
    """Make the y-axis be latitude."""
    ax.set_ylim([start_lat, end_lat])
    ax.set_yticks(np.arange(start_lat, end_lat + 1, 10))
    if start_lat == 0 and end_lat == 90:
        ax.set_yticklabels(['EQ', '', '', '30' + _DEGR, '', '',
                            '60' + _DEGR, '', '', '90' + _DEGR])
    elif start_lat == -90 and end_lat == 90:
        ax.set_yticklabels(['-90' + _DEGR, '', '', '-60' + _DEGR, '', '',
                            '-30' + _DEGR, '', '', 'EQ', '', '', '30' + _DEGR,
                            '', '', '60' + _DEGR, '', '', '90' + _DEGR])
    ax.set_ylabel(" ")


def facet_ax(width=4, cbar_mode=None, **kwargs):
    """Use faceted to create single panel figure."""
    if cbar_mode is None:
        fig, axarr = faceted(1, 1, width=width, **kwargs)
        return fig, axarr[0]
    else:
        fig, axarr, cax = faceted(1, 1, width=width,
                                  cbar_mode=cbar_mode, **kwargs)
        return fig, axarr[0], cax


def plot_lat_1d(arr, start_lat=-90, end_lat=90, sinlat=False,
                ax=None, lat_str=LAT_STR, ax_labels=False, **plot_kwargs):
    """Plot of the given array as a function of latitude."""
    arr_plot = arr.where((arr[lat_str] > start_lat) &
                         (arr[lat_str] < end_lat))
    if ax is None:
        ax = plt.gca()
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
    if ax is None:
        ax = plt.gca()
    ax.plot(lats[2:-2], arr[2:-2], **kwargs)


def panel_label(panel_num, ax=None, x=0.03, y=0.9, extra_text=None,
                **text_kwargs):
    if ax is None:
        ax = plt.gca()
    letters = 'abcdefghijklmnopqrstuvwxyz'
    label = '({})'.format(letters[panel_num])
    if extra_text is not None:
        label += ' {}'.format(extra_text)
    ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)


@default_gca
def mark_x0(ax=None, linewidth=0.5, color='0.5', x0=0, **kwargs):
    """Mark the x intercept line on the given axis."""
    return ax.axvline(x=x0, linewidth=linewidth, color=color, **kwargs)


@default_gca
def mark_y0(ax=None, linewidth=0.5, color='0.5', y0=0, **kwargs):
    """Mark the y intercept on the given axis."""
    return ax.axhline(y=y0, linewidth=linewidth, color=color, **kwargs)


@default_gca
def mark_one2one(ax=None, *line_args, linestyle=':', color='0.7',
                 linewidth=0.8, **line_kwargs):
    """Mark the identity line, y=x, on the given axis.

    From https://stackoverflow.com/a/28216751/1706640.

    """
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


def nb_savefig(name, fig=None, fig_dir="../figs", **kwargs):
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(fig_dir, name), **kwargs)


if __name__ == '__main__':
    pass
