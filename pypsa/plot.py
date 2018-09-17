

## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for plotting networks.
"""


# make the code as Python 3 compatible as possible
from __future__ import division
from __future__ import absolute_import
import six
from six import iteritems, string_types

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


plt_present = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.collections import LineCollection, PatchCollection
except:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except:
    basemap_present = False


pltly_present = True
try:
        import plotly.offline as pltly
except:
        pltly_present = False


def plot(network, margin=0.05, ax=None, basemap=True, bus_colors='grey',
         line_colors=None, bus_sizes=10, flow=None, line_widths=1, title="",
         line_cmap=None, bus_cmap=None, boundaries=None,
         geometry=False, branch_components=['Line', 'Link'], jitter=None):
    """
    Plot the network buses and lines using matplotlib and Basemap.

    Parameters
    ----------
    margin : float
        Margin at the sides as proportion of distance between max/min x,y
    ax : matplotlib ax, defaults to plt.gca()
        Axis to which to plot the network
    basemap : bool, default True
        Switch to use Basemap
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "b"
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 10
    line_colors : dict/pandas.Series
        Colors for the lines, defaults to "g" for Lines and "cyan" for
        Links. Colors for branches other than Lines can be
        specified using a pandas Series with a MultiIndex.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2. Widths for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    flow : snapshot/pandas.Series
        Flow to be displayed in the plot, defaults to None. If an element of
        network.snapshots is given, the flow at this timestamp will be
        displayed. Other flows can be specified by passing a pandas Serie with
        a MultiIndex.
    title : string
        Graph title
    line_cmap : plt.cm.ColorMap/str|dict
        If line_colors are floats, this color map will assign the colors.
        Use a dict to specify colormaps for more than one branch type.
    bus_cmap : plt.cm.ColorMap/str
        If bus_colors are floats, this color map will assign the colors
    boundaries : list of four floats
        Boundaries of the plot in format [x1,x2,y1,y2]
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses

    Returns
    -------
    bus_collection, branch_collection1, ... : tuple of Collections
        Collections for buses and branches.
    """

    defaults_for_branches = {
        'Link': dict(color="royalblue", width=1),
        'Line': dict(color="orange", width=1),
        'Transformer': dict(color='green', width=1)
    }
    if line_colors is None:
        line_colors = (pd.Series( defaults_for_branches)
                       .apply(pd.Series).loc[['Line','Link'], 'color']
                       .to_dict())

    if not plt_present:
        logger.error("Matplotlib is not present, so plotting won't work.")
        return

    if ax is None:
        ax = plt.gca()

    def compute_bbox_with_margins(margin, x, y):
        #set margins
        pos = np.asarray((x, y))
        minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
        xy1 = minxy - margin*(maxxy - minxy)
        xy2 = maxxy + margin*(maxxy - minxy)
        return tuple(xy1), tuple(xy2)

    x = network.buses["x"]
    y = network.buses["y"]

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    if basemap and basemap_present:
        resolution = 'l' if isinstance(basemap, bool) else basemap

        if boundaries is None:
            (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
        else:
            x1, x2, y1, y2 = boundaries
        bmap = Basemap(resolution=resolution, epsg=network.srid,
                       llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                       urcrnrlon=x2, ax=ax)
        bmap.drawcountries(linewidth=0.3, zorder=-1)
        bmap.drawcoastlines(linewidth=0.4, zorder=-1)

        x, y = bmap(x.values, y.values)
        x = pd.Series(x, network.buses.index)
        y = pd.Series(y, network.buses.index)

    if isinstance(bus_sizes, pd.Series) and isinstance(bus_sizes.index, pd.MultiIndex):
        # We are drawing pies to show all the different shares
        assert len(bus_sizes.index.levels[0].difference(network.buses.index)) == 0, \
            "The first MultiIndex level of bus_sizes must contain buses"
        assert isinstance(bus_colors, dict) and set(bus_colors).issuperset(bus_sizes.index.levels[1]), \
            "bus_colors must be a dictionary defining a color for each element " \
            "in the second MultiIndex level of bus_sizes"

        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)

        patches = []
        for b_i in bus_sizes.index.levels[0]:
            s = bus_sizes.loc[b_i]
            radius = s.sum()**0.5
            if radius == 0.0:
                ratios = s
            else:
                ratios = s/s.sum()

            start = 0.25
            for i, ratio in ratios.iteritems():
                patches.append(Wedge((x.at[b_i], y.at[b_i]), radius,
                                     360*start, 360*(start+ratio),
                                     facecolor=bus_colors[i]))
                start += ratio
        bus_collection = PatchCollection(patches, match_original=True)
        ax.add_collection(bus_collection)
    else:
        c = pd.Series(bus_colors, index=network.buses.index)
        s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)
        bus_collection = ax.scatter(x, y, c=c, s=s, cmap=bus_cmap, edgecolor='face')

    def as_branch_series(ser):
        if isinstance(ser, dict) and set(ser).issubset(branch_components):
            return pd.Series(ser)
        elif isinstance(ser, pd.Series):
            if isinstance(ser.index, pd.MultiIndex):
                return ser
            else:
                return pd.concat([ser.reindex(network.lines.index),
                                 ser.reindex(network.links.index)],
                                 keys=['Line', 'Link'], sort=True)
        else:
            index = pd.concat([network.lines, network.links],
                          keys=['Line', 'Link'], sort=True).index
            return pd.Series(ser, index)

    branch_collections = []

    line_colors = as_branch_series(line_colors)
    if flow is None:
        line_widths = as_branch_series(line_widths)
    else:
        if flow in network.snapshots:
            flow = pd.concat([network.lines_t.p0.loc[flow],
                          network.links_t.p0.loc[flow]], keys=['Line', 'Link'])

        # take line_widths as argument for scaling the arrows and linewidths
        assert ~(isinstance(line_widths, float) | isinstance(line_widths, int)
                | line_widths is None), """Setting flow and line_widths for each branch
                is not possible. For a given flow, the argument line_widths
                is restricted to a scaling factor only"""
        # set a rough estimate of the linescales which scales the size of the arrows
        # and lines for the default line_widths=2
        flow_scale = (len(network.lines)+100)**1.7 * 2./line_widths
        arrows = directed_flow(network, flow, ax=ax, flow_scale=flow_scale,
                               line_colors=line_colors)
        branch_collections.append(arrows)
        line_widths = as_branch_series(flow)/flow_scale
    if not isinstance(line_cmap, dict):
        line_cmap = {'Line': line_cmap}

    for c in network.iterate_components(branch_components):
        l_defaults = defaults_for_branches[c.name]
        l_widths = line_widths.get(c.name, l_defaults['width'])
        l_nums = None
        l_colors = line_colors.get(c.name, l_defaults['color'])

        if isinstance(l_colors, pd.Series):
            if issubclass(l_colors.dtype.type, np.number):
                l_nums = l_colors
                l_colors = None
            else:
                l_colors.fillna(l_defaults['color'], inplace=True)

        if not geometry:
            segments = (np.asarray(((c.df.bus0.map(x),
                                     c.df.bus0.map(y)),
                                    (c.df.bus1.map(x),
                                     c.df.bus1.map(y))))
                        .transpose(2, 0, 1))
        else:
            from shapely.wkt import loads
            from shapely.geometry import LineString
            linestrings = c.df.geometry.map(loads)
            assert all(isinstance(ls, LineString) for ls in linestrings), \
                "The WKT-encoded geometry in the 'geometry' column must be composed of LineStrings"
            segments = np.asarray(list(linestrings.map(np.asarray)))
            if basemap and basemap_present:
                segments = np.transpose(bmap(*np.transpose(segments, (2, 0, 1))), (1, 2, 0))

        l_collection = LineCollection(segments,
                                      linewidths=l_widths,
                                      antialiaseds=(1,),
                                      colors=l_colors,
                                      transOffset=ax.transData)

#        if annotate:
#            for line in c.df.index:
#                ax.annotate(line, ((x0[line]+x1[line])/2.,
#                                   (y0[line]+y1[line])/2.), size='small')

        if l_nums is not None:
            l_collection.set_array(np.asarray(l_nums))
            l_collection.set_cmap(line_cmap.get(c.name, None))
            l_collection.autoscale()

        ax.add_collection(l_collection)
        l_collection.set_zorder(1)

        branch_collections.append(l_collection)

    bus_collection.set_zorder(2)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()
    ax.axis('off')

    ax.set_title(title)

    return (bus_collection,) + tuple(branch_collections)


#This function was borne out of a breakout group at the October 2017
#Munich Open Energy Modelling Initiative Workshop to hack together a
#working example of plotly for networks, see:
#https://forum.openmod-initiative.org/t/breakout-group-on-visualising-networks-with-plotly/384/7

#We thank Bryn Pickering for holding the tutorial on plotly which
#inspired the breakout group and for contributing ideas to the iplot
#function below.

def iplot(network, fig=None, bus_colors='grey',
          bus_colorscale=None, bus_colorbar=None, bus_sizes=10, bus_text=None,
          line_colors=None, line_widths=2, line_text=None, title="",
          geoscope='europe', branch_components=['Line', 'Link'], iplot=True,
          jitter=None):
    """
    Plot the network buses and lines interactively using plotly.

    Parameters
    ----------
    fig : dict, default None
        If not None, figure is built upon this fig.
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "b"
    bus_colorscale : string
        Name of colorscale if bus_colors are floats, e.g. 'Jet', 'Viridis'
    bus_colorbar : dict
        Plotly colorbar, e.g. {'title' : 'my colorbar'}
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 10
    bus_text : dict/pandas.Series
        Text for each bus, defaults to bus names
    line_colors : dict/pandas.Series
        Colors for the lines, defaults to "g" for Lines and "cyan" for
        Links. Colors for branches other than Lines can be
        specified using a pandas Series with a MultiIndex.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2. Widths for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    line_text : dict/pandas.Series
        Text for lines, defaults to line names. Text for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    title : string
        Graph title
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    iplot : bool, default True
        Automatically do an interactive plot of the figure.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses

    Returns
    -------
    fig: dictionary for plotly figure
    """

    defaults_for_branches = {
        'Link': dict(color="royalblue", width=2),
        'Line': dict(color="orange", width=2),
        'Transformer': dict(color='green', width=2)
    }
    if line_colors is None:
        line_colors = (pd.Series( defaults_for_branches)
                       .apply(pd.Series).loc[['Line','Link'], 'color']
                       .to_dict())


    if fig is None:
        fig = dict(data=[],layout={})

    if bus_text is None:
        bus_text = 'Bus ' + network.buses.index

    x = network.buses.x
    y = network.buses.y

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    bus_trace = dict(lon=x, lat=y,
                     text=bus_text,
                     type="scattergeo",
                     mode="markers",
                     hoverinfo="text",
                     marker=dict(color=bus_colors,
                                 size=bus_sizes)
                     )

    if bus_colorscale is not None:
        bus_trace['marker']['colorscale'] = bus_colorscale

    if bus_colorbar is not None:
        bus_trace['marker']['colorbar'] = bus_colorbar


    def as_branch_series(ser):
        if isinstance(ser, dict) and set(ser).issubset(branch_components):
            return pd.Series(ser)
        elif isinstance(ser, pd.Series):
            if isinstance(ser.index, pd.MultiIndex):
                return ser
            index = ser.index
            ser = ser.values
        else:
            index = network.lines.index
        return pd.Series(ser,
                         index=pd.MultiIndex(levels=(["Line"], index),
                                             labels=(np.zeros(len(index)),
                                                     np.arange(len(index)))))

    line_colors = as_branch_series(line_colors)
    line_widths = as_branch_series(line_widths)

    if line_text is not None:
        line_text = as_branch_series(line_text)

    shapes = []

    shape_traces = []

    for c in network.iterate_components(branch_components):
        l_defaults = defaults_for_branches[c.name]
        l_widths = line_widths.get(c.name, l_defaults['width'])
        l_colors = line_colors.get(c.name, l_defaults['color'])
        l_nums = None

        if line_text is None:
            l_text = c.name + ' ' + c.df.index
        else:
            l_text = line_text.get(c.name)

        if isinstance(l_colors, pd.Series):
            if issubclass(l_colors.dtype.type, np.number):
                l_nums = l_colors
                l_colors = None
            else:
                l_colors.fillna(l_defaults['color'], inplace=True)

        x0 = c.df.bus0.map(x)
        x1 = c.df.bus1.map(x)

        y0 = c.df.bus0.map(y)
        y1 = c.df.bus1.map(y)

        for line in c.df.index:
            color = l_colors if isinstance(l_colors, string_types) else l_colors[line]
            width = l_widths if isinstance(l_widths, (int, float)) else l_widths[line]


            shapes.append(dict(type='scattergeo',
                               mode='lines',
                               lon=[x0[line], x1[line]],
                               lat=[y0[line], y1[line]],
#                               x0=x0[line],
#                               y0=y0[line],
#                               x1=x1[line],
#                               y1=y1[line],
                               opacity=0.7,
                               line=dict(color=color, width=width)))


        shape_traces.append(dict(lon=0.5*(x0+x1),
                                 lat=0.5*(y0+y1),
                                 text=l_text,
                                 type="scattergeo",
                                 mode="markers",
                                 hoverinfo="text",
                                 marker=dict(opacity=0.)))

    fig['data'].extend(shapes+shape_traces+[bus_trace])

    fig['layout'].update(dict(title=title,
                              hovermode='closest',
                              showlegend=False,
                              geo = dict(
                                scope=geoscope,
                                projection=dict( type='azimuthal equal area' ),
                                showland = True,
                                landcolor = 'rgb(243, 243, 243)',
                                countrycolor = 'rgb(204, 204, 204)',
                                )
                              ))


    if iplot:
        if not pltly_present:
            logger.warning("Plotly is not present, so interactive plotting won't work.")
        else:
            pltly.iplot(fig)

    return fig



def directed_flow(n, flow, flow_scale=None, ax=None, line_colors='darkgreen'):
#    this funtion is used for diplaying arrows representing the network flow
    from matplotlib.patches import FancyArrow
    if flow_scale is None:
        flow_scale = 1
    if ax is None:
        ax = plt.gca()
#    set the scale of the arrowsizes
    arrowsize = (flow.abs()
                .pipe(lambda ds: np.sqrt(ds/flow_scale))
                .clip(lower=1e-8))
    fdata = pd.concat(
            [pd.DataFrame({'x1': getattr(n, l).bus0.map(n.buses.x),
                          'y1': getattr(n, l).bus0.map(n.buses.y),
                          'x2': getattr(n, l).bus1.map(n.buses.x),
                          'y2': getattr(n, l).bus1.map(n.buses.y),
                          # make area not width proportional to flow
                          'arrowsize': arrowsize.loc[l[:-1].capitalize()]
                          .reindex((getattr(n, l).index)),
                          'direction': np.sign(flow).loc[l[:-1].capitalize()]
                          .reindex((getattr(n, l).index))})
            for l in ['lines', 'links']],
            keys=['Line', 'Link'])
    fdata['linelength'] = (np.sqrt((fdata.x1-fdata.x2)**2.+
                                        (fdata.y1 - fdata.y2)**2))
    fdata['arrowtolarge'] = (1.5 * fdata.arrowsize
                                            > fdata.loc[:, 'linelength'])

    #swap coords for negativ directions
    fdata.loc[fdata.direction==-1., ['x1', 'x2', 'y1', 'y2']] = (
        fdata.loc[fdata.direction==-1., ['x2', 'x1', 'y2', 'y1']].values)

    fdata['arrows'] = (
            fdata[(fdata.linelength>0.)&(~fdata.arrowtolarge)]
                .apply(lambda ds:
                    FancyArrow(ds.x1, ds.y1,
                               0.6*(ds.x2 - ds.x1)-ds.arrowsize
                                   *0.75*(ds.x2 - ds.x1)/ds.linelength,
                               0.6*(ds.y2 - ds.y1)-ds.arrowsize
                                   *0.75*(ds.y2 - ds.y1)/ds.linelength,
                               head_width=ds.arrowsize
                               ), axis=1) )
    fdata.loc[(fdata.linelength>0.)&(fdata.arrowtolarge), 'arrows']= (
            fdata[(fdata.linelength>0.)&(fdata.arrowtolarge)]
                .apply(lambda ds:
                    FancyArrow(ds.x1, ds.y1,
                               0.001*(ds.x2 - ds.x1),
                               0.001*(ds.y2 - ds.y1),
                               head_width=ds.arrowsize
                               ), axis=1) )
    if not isinstance(line_colors.index, pd.MultiIndex):
        line_colors = (line_colors.reindex(fdata.index.get_level_values(0))
                        .set_axis(fdata.index, inplace=False))
    fdata = fdata.assign(color=line_colors)
    arrowcol = PatchCollection(fdata[fdata.arrows.notnull()].arrows,
                              color=fdata.color,
                              edgecolors='k',
                              linewidths=0.,
                              zorder=2, alpha=1)
    ax.add_collection(arrowcol)
    return arrowcol