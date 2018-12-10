

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
__copyright__ = ("Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), "
                 "GNU GPL 3")


plt_present = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.lines import Line2D
except:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    basemap_present = False


cartopy_present = True
try:
    import cartopy
    import cartopy.crs as ccrs
except:
    cartopy_present = False


pltly_present = True
try:
        import plotly.offline as pltly
except ImportError:
        pltly_present = False

idx = pd.IndexSlice


branch_defaults = pd.DataFrame({
    'Link': {'color': "teal", 'width': 1},
    'Line': {'color': "gold", "width": 1},
    'Transformer': {'color': 'forestgreen', 'width': 1}})


def plot(n, margin=0.05, ax=None, basemap=True, bus_colors='grey',
         line_colors=None, bus_sizes=10, flow=None, generation=None,
         line_widths=1, title="", line_cmap=None, bus_cmap=None,
         boundaries=None, geometry=False, legend=True,
         branch_components=['Line', 'Link'], jitter=None):
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
    flow : snapshot/pandas.Series/function/string
        Flow to be displayed in the plot, defaults to None. If an element of
        network.snapshots is given, the flow at this timestamp will be
        displayed. If an aggregation function is given, is will be applied
        to the total network flow via pandas.DataFrame.agg (accepts also
        function names). Otherwise flows can be specified by passing a pandas
        Series with MultiIndex.
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
    # 1. doublecheck plotting packages
    # 2. set bus_sizes from generation argument
    # 3. plot bus_sizes
    # 4. plot flow arrows if flow argument is present
    # 5. plot lines

    # 1. doublecheck plotting packages
    if not plt_present:
        logger.error("Matplotlib is not present, so plotting won't work.")
        return

    if ax is None:
        if cartopy_present and basemap:
            ax = plt.gca(projection=ccrs.PlateCarree())
        else:
            ax = plt.gca()
    else:
        if cartopy_present and basemap:
            import cartopy.mpl.geoaxes
            assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot), (
                    'The passed axis is not a GeoAxesSubplot. You can '
                    'create one with: \nimport cartopy.crs as ccrs \n'
                    'fig, ax = plt.subplots('
                    'subplot_kw={"projection":ccrs.PlateCarree()})')

    if basemap:
        x, y = draw_map(n, jitter, ax, boundaries, margin, basemap)
    else:
        x, y = n.buses['x'], n.buses['y']

    # 2. set bus_sizes from generation argument
    if isinstance(generation, str) or callable(generation):
        bus_sizes = (n.generators
                     .assign(sizes = n.generators_t.p.agg(generation))
                     .groupby(['bus' , 'carrier'])['sizes'].sum()
                     [lambda ds: ds>0] * bus_sizes / 1e3)
    elif generation in n.snapshots:
        bus_sizes = (n.generators
                     .assign(sizes = n.generators_t.p.loc[generation])
                     .groupby(['bus' , 'carrier'])['sizes'].sum()
                     [lambda ds: ds>0] * bus_sizes / 1e3)

    # 3. plot bus_sizes
    if isinstance(bus_sizes, pd.Series) and isinstance(bus_sizes.index,
                 pd.MultiIndex):
        # We are drawing pies to show all the different shares
        assert bus_sizes.index.levels[0].difference(n.buses.index).empty,\
            "The first MultiIndex level of bus_sizes must contain buses"
        bus_colors = pd.Series(bus_colors).reindex(bus_sizes.index.levels[1])

        #make sure that all carriers get colors
        if bus_colors.isnull().any():
            from matplotlib.colors import cnames
            missing_colors_i = bus_colors[bus_colors.isnull()].index
            missing_colors = pd.Series(
                    pd.Series(cnames).sample(len(missing_colors_i)).index,
                    index=missing_colors_i)
            bus_colors = bus_colors.append(missing_colors).dropna()
            logger.warning("Colors for carriers {} not defined, setting"
                           " them randomly to: {}"
                           .format(list(missing_colors_i),
                                   missing_colors.to_dict()))
        if legend:
            handles = marker_from_color_series(bus_colors)
            labels = bus_colors.index.tolist()
            ax.add_artist(
                ax.legend(handles=handles, edgecolor='w',
                          facecolor='inherit', fancybox=True,
                          labels=labels,
                          loc=2, framealpha=1))

        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)

        patches = []
        for b_i in bus_sizes.index.unique(0):
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
        c = pd.Series(bus_colors, index=n.buses.index)
        s = pd.Series(bus_sizes, index=n.buses.index, dtype="float").fillna(0)
        bus_collection = ax.scatter(x, y, c=c, s=s, cmap=bus_cmap,
                                    edgecolor='face')

    # 4. plot flow arrows if flow argument is present
    branch_collections = []
#    l_default = (None if pd.api.types.is_numeric_dtype(line_colors)
#                 else branch_defaults.loc['color'])
    line_colors = as_series(line_colors, n, branch_components,
                            branch_defaults.loc['color'])

    line_widths = as_series(line_widths, n, branch_components,
                            branch_defaults.loc['width'])
    if flow in n.snapshots:
        flow = (pd.concat([n.pnl(c).p0.loc[flow]
                for c in branch_components],
                keys=branch_components, sort=True))
    elif isinstance(flow, str) or callable(flow):
        flow = (pd.concat([n.pnl(c).p0 for c in branch_components],
                axis=1, keys=branch_components, sort=True)
                .agg(flow, axis=0))
    if flow is not None:
        # set a rough estimate of flow_scale
        flow_scale = (len(n.lines)+100) / line_widths
        arrows = directed_flow(n, flow, ax=ax, flow_scale=flow_scale,
                               line_colors=line_colors)
        branch_collections.append(arrows)
        line_widths = (as_series(flow, n, branch_components,
                       branch_defaults.loc['width']) / flow_scale)

    if not isinstance(line_cmap, dict):
        line_cmap = {'Line': line_cmap}

    # 5. plot lines
    for c in n.iterate_components(branch_components):
        l_widths = line_widths.loc[c.name]
        l_nums = None
        l_colors = line_colors.loc[c.name]

        if pd.api.types.is_numeric_dtype(l_colors):
            l_nums = l_colors
            l_colors = None

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
            assert all(isinstance(ls, LineString) for ls in linestrings), (
                "The WKT-encoded geometry in the 'geometry' column must "
                "be composed of LineStrings")
            segments = np.asarray(list(linestrings.map(np.asarray)))
            if basemap and basemap_present:
                segments = np.transpose(bmap(*np.transpose(
                        segments, (2, 0, 1))), (1, 2, 0))

        l_collection = LineCollection(segments,
                                      linewidths=l_widths,
                                      antialiaseds=(1,),
                                      colors=l_colors,
                                      transOffset=ax.transData)

        if l_nums is not None:
            l_collection.set_array(np.asarray(l_nums))
            l_collection.set_cmap(line_cmap.get(c.name, None))
            l_collection.autoscale()

        ax.add_collection(l_collection)
        l_collection.set_zorder(1)

        branch_collections.append(l_collection)

    bus_collection.set_zorder(3)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()
    if basemap_present:
        ax.axis('off')
    if cartopy_present:
        ax.outline_patch.set_visible(False)

    ax.set_title(title)

    return (bus_collection,) + tuple(branch_collections)



def marker_from_color_series(ds):
    return (ds.apply(lambda x:
            Line2D([], [], c=x, marker='.', linestyle='None', markersize=20))
            .tolist())


def as_series(ser, n, components, default=np.nan):
    index = n.branches().loc[idx[components, :], :].index
    if not isinstance(ser, (pd.Series, dict)):
        ser = pd.Series(ser, index=index)
    elif isinstance(ser, pd.Series) & isinstance(ser.index, pd.MultiIndex):
        ser = ser.reindex(index)
    else:
        ser = (pd.concat([pd.Series(ser).reindex(index, level=0).dropna(),
                         pd.Series(ser).reindex(index, level=1).dropna()],
                         sort=False)[lambda ds: ~ds.index.duplicated()]
               .reindex(index))
    if isinstance(default, (dict, pd.Series)):
        default = pd.Series(default).reindex(index, level=0)
    return ser.where(ser.notnull(), default)


def compute_bbox_with_margins(margin, x, y):
    # set margins
    pos = np.asarray((x, y))
    minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
    xy1 = minxy - margin*(maxxy - minxy)
    xy2 = maxxy + margin*(maxxy - minxy)
    return tuple(xy1), tuple(xy2)


def draw_map(network=None, jitter=None, ax=None, boundaries=None,
             margin=0.05, basemap=True):

    x = network.buses["x"]
    y = network.buses["y"]

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    if boundaries is None:
        (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
    else:
        x1, x2, y1, y2 = boundaries

    if basemap_present:
        resolution = 'l' if isinstance(basemap, bool) else basemap
        bmap = Basemap(resolution=resolution, epsg=network.srid,
                       llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                       urcrnrlon=x2, ax=ax)
        bmap.drawcountries(linewidth=0.3, zorder=-1)
        bmap.drawcoastlines(linewidth=0.4, zorder=-1)

        x, y = bmap(x.values, y.values)
        x = pd.Series(x, network.buses.index)
        y = pd.Series(y, network.buses.index)

    if cartopy_present:
        resolution = '50m' if isinstance(basemap, bool) else basemap
        assert resolution in ['10m', '50m', '110m'], (
                "Resolution has to be one of '10m', '50m', '110m'")
        ax.set_extent([x1, x2, y1, y2], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, zorder=-1, resolution=resolution)
        border = cartopy.feature.BORDERS.with_scale(resolution)
        ax.add_feature(border, linewidth=0.3)

    return x, y


def directed_flow(n, flow, flow_scale=None, ax=None, line_colors='darkgreen',
                  branch_comps=['Line', 'Link']):
    """
    Helper function to generate arrows from flow data.
    """
#    this funtion is used for diplaying arrows representing the network flow
    from matplotlib.patches import FancyArrow
    if flow_scale is None:
        flow_scale = 1
    if ax is None:
        ax = plt.gca()
    # align index level names
    flow_scale.index.names = flow.index.names
#    set the scale of the arrowsizes
    fdata = pd.concat([pd.DataFrame(
                      {'x1': n.df(l).bus0.map(n.buses.x),
                       'y1': n.df(l).bus0.map(n.buses.y),
                       'x2': n.df(l).bus1.map(n.buses.x),
                       'y2': n.df(l).bus1.map(n.buses.y)})
                      for l in branch_comps], keys=branch_comps)
    fdata['arrowsize'] = (flow.abs()
                          .pipe(lambda ds: np.sqrt(ds / flow_scale))
                          .clip(lower=1e-8) / 2.)
    fdata['direction'] = np.sign(flow)
    fdata['linelength'] = (np.sqrt((fdata.x1 - fdata.x2)**2. +
                           (fdata.y1 - fdata.y2)**2))
    fdata['arrowtolarge'] = (1.5 * fdata.arrowsize >
                             fdata.loc[:, 'linelength'])

    # swap coords for negativ directions
    fdata.loc[fdata.direction == -1., ['x1', 'x2', 'y1', 'y2']] = \
        fdata.loc[fdata.direction == -1., ['x2', 'x1', 'y2', 'y1']].values

    fdata['arrows'] = (
            fdata[(fdata.linelength > 0.) & (~fdata.arrowtolarge)]
            .apply(lambda ds:
                   FancyArrow(ds.x1, ds.y1,
                              0.6*(ds.x2 - ds.x1) - ds.arrowsize
                              * 0.75 * (ds.x2 - ds.x1) / ds.linelength,
                              0.6 * (ds.y2 - ds.y1) - ds.arrowsize
                              * 0.75 * (ds.y2 - ds.y1)/ds.linelength,
                              head_width=ds.arrowsize), axis=1))
    fdata.loc[(fdata.linelength > 0.) & (fdata.arrowtolarge), 'arrows'] = \
        (fdata[(fdata.linelength > 0.) & (fdata.arrowtolarge)]
         .apply(lambda ds:
                FancyArrow(ds.x1, ds.y1,
                           0.001*(ds.x2 - ds.x1),
                           0.001*(ds.y2 - ds.y1),
                           head_width=ds.arrowsize), axis=1))
    fdata = fdata.assign(color=line_colors)
    arrowcol = PatchCollection(fdata[fdata.arrows.notnull()].arrows,
                               color=fdata.color,
                               edgecolors='k',
                               linewidths=0.,
                               zorder=2, alpha=1)
    ax.add_collection(arrowcol)
    return arrowcol


#This function was born out of a breakout group at the October 2017
#Munich Open Energy Modelling Initiative Workshop to hack together a
#working example of plotly for networks, see:
#https://forum.openmod-initiative.org/t/ \
#breakout-group-on-visualising-networks-with-plotly/384/7

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

    line_colors = as_series(line_colors, network, branch_components,
                            branch_defaults.loc['color'])
    line_widths = as_series(line_widths, network, branch_components,
                            branch_defaults.loc['width'])

    if line_text is not None:
        line_text = as_series(line_text, network, branch_components)

    shapes = []
    shape_traces = []

    for c in network.iterate_components(branch_components):
        l_widths = line_widths.loc[c.name]
        l_colors = line_colors.loc[c.name]

        if line_text is None:
            l_text = c.name + ' ' + c.df.index
        else:
            l_text = line_text.get(c.name)

        x0 = c.df.bus0.map(x)
        x1 = c.df.bus1.map(x)

        y0 = c.df.bus0.map(y)
        y1 = c.df.bus1.map(y)

        for line in c.df.index:
            color = l_colors[line]
            width = l_widths[line]

            shapes.append(dict(type='scattergeo',
                               mode='lines',
                               lon=[x0[line], x1[line]],
                               lat=[y0[line], y1[line]],
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
                                countrycolor = 'rgb(204, 204, 204)')
                              ))


    if iplot:
        if not pltly_present:
            logger.warning("Plotly is not present, so interactive "
                           "plotting won't work.")
        else:
            pltly.iplot(fig)

    return fig



