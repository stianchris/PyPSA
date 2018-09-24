#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:14:49 2018

@author: fabian
"""

# This side-package is created for use as flow and cost allocation.

from .pf import calculate_PTDF
from numpy.linalg import inv
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


# %% utility functions


def pinv(df):
    return pd.DataFrame(np.linalg.pinv(df), df.columns, df.index)


def diag(df):
    """
    Convenience function to select diagonal from a square matrix, or to build
    a diagonal matrix from a series.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
    """
    if isinstance(df, pd.DataFrame):
        if len(df.columns) == len(df.index) > 1:
            return pd.DataFrame(np.diagflat(np.diag(df)), df.index, df.columns)
    return pd.DataFrame(np.diagflat(df.values),
                        index=df.index, columns=df.index)


def incidence_matrix(n, branch_components=['Link', 'Line']):
    buses = n.buses.index
    K = []
    for c in n.iterate_components(branch_components):
        K.append((c.df.assign(K=1).set_index('bus0', append=True)['K']
                 .unstack().reindex(columns=buses).fillna(0)
                 - c.df.assign(K=1).set_index('bus1', append=True)['K']
                 .unstack().reindex(columns=buses).fillna(0)).T)
    return pd.concat(K, keys=branch_components, axis=1)


def PTDF(n, branch_components={'Line'}, snapshot=None, scale=1.):
    n.calculate_dependent_values()
    n.lines = n.lines.assign(carrier=n.lines.bus0.map(n.buses.carrier))
    assert (n.lines.carrier == n.lines.bus1.map(n.buses.carrier)).all()
    K = incidence_matrix(n, branch_components)
    omega = []
    for c in branch_components:
        if c == 'Line':
            (omega.append(1/(n.lines.x_pu.where(n.lines.carrier == 'AC', 0)
             + n.lines.r_pu.where(n.lines.carrier == 'DC', 0))))
        if c == 'Link':
            if snapshot is None:
                logger.warn('Link in argument "branch_components", but no '
                            'snapshot given. Falling back to first snapshot')
                snapshot = n.snapshots[0]
            elif isinstance(snapshot, pd.DatetimeIndex):
                snapshot = snapshot[0]
            omega.append(n.links_t.p0.loc[snapshot].T *scale)
    Omega = diag(pd.concat(omega, keys=branch_components))
    return Omega.dot(K.T).dot(pinv(K.dot(Omega).dot(K.T))), pd.concat(omega, keys=branch_components)


def network_injection(n, snapshots=None):
    """
    Function to determine the total network injection including passive and
    active branches.
    """
    if snapshots is None:
        snapshots = n.snapshots
    if isinstance(snapshots, pd.Timestamp):
        snapshots = [snapshots]
    return (pd.concat({c.name:
            c.pnl.p.multiply(c.df.sign, axis=1)
            .groupby(c.df.bus, axis=1).sum()
            for c in n.iterate_components(n.controllable_one_port_components)},
            sort=True)
            .sum(level=1)
            .reindex(index=snapshots, columns=n.buses_t.p.columns,
                     fill_value=0)).T


def is_balanced(n, tol=1e-9):
    """
    Helper function to double check whether network flow is balanced
    """
    K = incidence_matrix(n)
    F = pd.concat([n.lines_t.p0, n.links_t.p0], axis=1,
                  keys=['Line', 'Link']).T
    return (K.dotF).sum(0).max() < tol

#%%

def average_participation(n, snapshot, per_bus=False, normalized=False,
                          downstream=True):
    """
    Allocate the network flow in according to the method 'Average
    participation' or 'Flow tracing' firstly presented in [1,2].
    The algorithm itself is derived from [3]. The general idea is to
    follow active power flow from source to sink (or sink to source)
    using the principle of proportional sharing and calculate the
    partial flows on each line, or to each bus where the power goes
    to (or comes from).

    This method provdes two general options:
        Downstream:
            The flow of each nodal power injection is traced through
            the network and decomposed the to set of lines/buses
            on which is flows on/to.
        Upstream:
            The flow of each nodal power demand is traced
            (in reverse direction) through the network and decomposed
            to the set of lines/buses where it comes from.

    [1] J. Bialek, “Tracing the flow of electricity,”
        IEE Proceedings - Generation, Transmission and Distribution,
        vol. 143, no. 4, p. 313, 1996.
    [2] D. Kirschen, R. Allan, G. Strbac, Contributions of individual
        generators to loads and flows, Power Systems, IEEE
        Transactions on 12 (1) (1997) 52–60. doi:10.1109/59.574923.
    [3] J. Hörsch, M. Schäfer, S. Becker, S. Schramm, and M. Greiner,
        “Flow tracing as a tool set for the analysis of networked
        large-scale renewable electricity systems,” International
        Journal of Electrical Power & Energy Systems,
        vol. 96, pp. 390–397, Mar. 2018.



    Parameters
    ----------
    network : pypsa.Network() object with calculated flow data

    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow
    downstream : Boolean, default True
        Whether to use downstream or upstream method.

    """

    n.calculate_dependent_values()
    buses = n.buses.index
    lines = n.lines.index
    links = n.links.index
    f_in = (n.lines_t.p0.loc[[snapshot]].T
             .append(n.links_t.p0.loc[[snapshot]].T))
    f_out = (-n.lines_t.p1.loc[[snapshot]].T
             .append(n.links_t.p1.loc[[snapshot]].T))
    p = n.buses_t.p.loc[[snapshot]].T
    K = pd.DataFrame(n.incidence_matrix(branch_components={'Line', 'Link'},
                                        busorder=buses).todense(),
                     index=buses, columns=lines.append(links))
#    set incidence matrix in direction of flow
    K = K.mul(np.sign(f_in[snapshot]))
    f_in = f_in.abs()
    f_out = f_out.abs()

    n_in = K.clip_upper(0).dot(f_in).abs()  # network inflow
    p_in = p.clip_lower(0)  # nodal inflow
    p_out = p.clip_upper(0).abs()  # nodal outflow

    F_in = (p_in + n_in)[snapshot]
    nzero_i = F_in[F_in != 0].index
#   F_in = F_out = (p_out + n_out)[snapshot]

    # flows from bus (index) to bus (columns):
    F = (K.dot(diag(f_in)).clip_lower(0).dot(K.T).clip_upper(0).abs())

    # upstream
    Q = (pd.DataFrame(inv((diag(F_in) - F.T).loc[nzero_i, nzero_i])
                      .dot(diag(p_in.reindex(nzero_i))),
                      index=nzero_i, columns=nzero_i)
         .reindex(index=buses, columns=buses, fill_value=0))
    # downstream
    R = (pd.DataFrame(inv((diag(F_in) - F).loc[nzero_i, nzero_i])
                      .dot(diag(p_out.reindex(nzero_i))),
                      index=nzero_i, columns=nzero_i)
         .reindex(index=buses, columns=buses, fill_value=0))

    #  equation (12)
    #  (Q.T.dot(diag(p_out)).T == R.T.dot(diag(p_in)) ).all().all()

    if per_bus:
        if not normalized:
            Q = Q.mul(p_out[snapshot], axis=0)
        Q = (Q.rename_axis('bus/sink').rename_axis('bus/source', axis=1)
             .stack()[lambda ds: ds != 0])
        if downstream:
            T = Q.swaplevel(0).sort_index()
        else:
            #  equal to weighting and stacking (from above) with R:
            T = Q

#    create artificial injection patterns following the flow trace
    else:
        if downstream:
            T = Q
            weight = (p_in + p_out)[snapshot]
            ref_b = p_in[snapshot] == 0
            tag = '/source'
        else:
            T = R
            weight = F_in
            ref_b = p_out[snapshot] == 0
            tag = '/sink'

        T = T.mul((weight), axis=0)
        T = diag(T) - T[ref_b].reindex_like(T, fill_value=0)
        T = PTDF(n, branch_components=['Link', 'Line']).dot(T).round(10)
        if normalized:
            T.div(f_in[snapshot], axis=0)
        T = T.rename_axis('line').rename_axis('bus' + tag, axis=1).T
        T = T.stack()[lambda ds: ds != 0]
    return pd.concat([T], keys=[snapshot], names=['snapshots'])


def marginal_participation(n, snapshot, q=0.5, normalized=False,
                           per_bus=False):
    '''
    Allocate line flows according to linear sensitvities of nodal power
    injection given by the changes in the power transfer distribution
    factors (PTDF)[1-3]. As the method is based on the DC-approximation,
    it works on subnetworks only as link flows are not taken into account.
    Note that this method does not exclude counter flows.

    [1] F. J. Rubio-Oderiz, I. J. Perez-Arriaga, Marginal pricing of
        transmission services: a comparative analysis of network cost
        allocation methods, IEEE Transactions on Power Systems 15 (1)
        (2000) 448–454. doi:10.1109/59.852158.
    [2] M. Schäfer, B. Tranberg, S. Hempel, S. Schramm, M. Greiner,
        Decompositions of injection patterns for nodal flow allocation
        in renewable electricity networks, The European Physical
        Journal B 90 (8) (2017) 144.
    [3] T. Brown, “Transmission network loading in Europe with high
        shares of renewables,” IET Renewable Power Generation,
        vol. 9, no. 1, pp. 57–65, Jan. 2015.


    Parameters
    ----------
    network : pypsa.Network() object with calculated flow data

    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    q : float, default 0.5
        split between net producers and net consumers.
        If q is zero, only the impact of net load is taken into
        account. If q is one, only net generators are taken
        into account.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow

    '''
    H = PTDF(n)
    p = n.buses_t.p.loc[snapshot]
    p_plus = p.clip_lower(0)
    f = n.lines_t.p0.loc[snapshot]
#   unbalanced flow from positive injection:
    f_plus = H.dot(p_plus)
    k_plus = (q*f - f_plus)/p_plus.sum()
    if normalized:
        Q = H.add(k_plus, axis=0).mul(p, axis=1).div(f, axis=0).round(10).T
    else:
        Q = H.add(k_plus, axis=0).mul(p, axis=1).round(10).T
    if per_bus:
        K = incidence_matrix(n)
        Q = K.dot(Q.T)
        Q = (Q.rename_axis('bus').rename_axis('bus', axis=1)
             .stack().round(8)[lambda ds:ds != 0])
    else:
        Q = (Q.rename_axis('bus').rename_axis("line", axis=1)
             .stack().round(8)[lambda ds:ds != 0])
    return pd.concat([Q], keys=[snapshot], names=['snapshots'])


def virtual_injection_pattern(n, snapshot, normalized=False, per_bus=False,
                              downstream=True):
    """
    Sequentially calculate the load flow induced by individual
    power sources in the network ignoring other sources and scaling
    down sinks. The sum of the resulting flow of those virtual
    injection patters is the total network flow. This method matches
    the 'Marginal participation' method with q = 1.



    Parameters
    ----------
    network : pypsa.Network object with calculated flow data
    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow

    """

    H = PTDF(n)
    p = n.buses_t.p.loc[snapshot]
    p_plus = p.clip_lower(0)
    p_minus = p.clip_upper(0)
    f = n.lines_t.p0.loc[snapshot]
    if downstream:
        indiag = diag(p_plus)
        offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T)
                   .div(p_plus.sum()))
#                   .pipe(lambda df: df - np.diagflat(np.diag(df)))
    else:
        indiag = diag(p_minus)
        offdiag = (p_plus.to_frame().dot(p_minus.to_frame().T)
                   .div(p_minus.sum()))
#                   .pipe(lambda df: df - np.diagflat(np.diag(df))))
    vip = indiag + offdiag
    if per_bus:
        Q = (vip[indiag.sum() == 0].T
             .rename_axis('bus/sink', axis=int(downstream))
             .rename_axis('bus/source', axis=int(not downstream))
             .stack()[lambda ds:ds != 0]).abs()
#        switch to counter stream by Q.swaplevel(0).sort_index()
    else:
        Q = H.dot(vip).round(10).T
        if normalized:
            # normalized colorvectors
            Q /= f
        Q = (Q.rename_axis('bus').rename_axis("line", axis=1)
             .stack().round(8)[lambda ds: ds != 0])
    return pd.concat([Q], keys=[snapshot], names=['snapshots'])


def optimal_flow_shares(n, snapshot, method='min', downstream=True,
                        per_bus=False, **kwargs):
    """



    """
    from scipy.optimize import minimize
    H = PTDF(n)
    p = n.buses_t.p.loc[snapshot]
    p_plus = p.clip_lower(0)
    p_minus = p.clip_upper(0)
    pp = p.to_frame().dot(p.to_frame().T).div(p).fillna(0)
    if downstream:
        indiag = diag(p_plus)
        offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T)
                   .div(p_plus.sum()))
        pp = pp.clip_upper(0).add(diag(pp)).mul(np.sign(p.clip_lower(0)))
        bounds = pd.concat([pp.stack(), pp.stack().clip_lower(0)], axis=1,
                           keys=['lb', 'ub'])

#                   .pipe(lambda df: df - np.diagflat(np.diag(df)))
    else:
        indiag = diag(p_minus)
        offdiag = (p_plus.to_frame().dot(p_minus.to_frame().T)
                   .div(p_minus.sum()))
        pp = pp.clip_lower(0).add(diag(pp)).mul(-np.sign(p.clip_upper(0)))
        bounds = pd.concat([pp.stack().clip_upper(0), pp.stack()], axis=1,
                           keys=['lb', 'ub'])
    x0 = (indiag + offdiag).stack()
    N = len(n.buses)
    if method == 'min':
        sign = 1
    elif method == 'max':
        sign = -1

    def minimization(df):
        return sign * (H.dot(df.reshape(N, N)).stack()**2).sum()

    constr = [
            #   nodal balancing
            {'type': 'eq', 'fun': lambda df: df.reshape(N, N).sum(0)},
            #    total injection of colors
            {'type': 'eq', 'fun': lambda df: df.reshape(N, N).sum(1)-p.values}
            ]

    #   sources-sinks-fixation
    res = minimize(minimization, x0, constraints=constr,
                   bounds=bounds, options={'maxiter': 1000}, tol=1e-5,
                   method='SLSQP')
    print(res)
    sol = pd.DataFrame(res.x.reshape(N, N), columns=n.buses.index,
                       index=n.buses.index).round(10)
    if per_bus:
        return (sol[indiag.sum()==0].T
                .rename_axis('bus/sink', axis=int(downstream))
                .rename_axis('bus/source', axis=int(not downstream))
                .stack()[lambda ds:ds != 0])
    else:
        return H.dot(sol).round(8)


def zbus_transmission(n, snapshot):
    '''
    This allocation builds up on the method presented in [1]. However, we
    provide for non-linear power flow an additional DC-approximated
    modification, neglecting the series resistance r for lines.


    [1] A. J. Conejo, J. Contreras, D. A. Lima, and A. Padilha-Feltrin,
        “$Z_{\rm bus}$ Transmission Network Cost Allocation,” IEEE Transactions
        on Power Systems, vol. 22, no. 1, pp. 342–349, Feb. 2007.

    '''
    n.calculate_dependent_values()
    buses = n.buses.index
    slackbus = n.sub_networks.obj[0].slack_bus


    # linearised method, start from linearised admittance matrix
    Y_diag = diag((1/n.lines.x).groupby(n.lines.bus0).sum()
                  .reindex(n.buses.index, fill_value=0)) \
             + diag((1/n.lines.x).groupby(n.lines.bus1).sum()
                    .reindex(n.buses.index, fill_value=0))
    Y_offdiag = (1/n.lines.set_index(['bus0', 'bus1']).x
                 .unstack().reindex(index=buses, columns=buses)).fillna(0)
    Y = Y_diag - Y_offdiag - Y_offdiag.T

    Z = pd.DataFrame(pinv(Y), buses, buses)
    # set angle of slackbus to 0
    Z = Z.add(-Z.loc[slackbus])
    # DC-approximated S = P
    S = n.buses_t.p.loc[[snapshot]].T
    V = n.buses.v_nom.to_frame(snapshot) * n.buses_t.v_ang.T
    I = Y.dot(V)

    # -------------------------------------------------------------------------
    # nonlinear method start with full admittance matrix from pypsa
    n.sub_networks.obj[0].calculate_Y()
    # Zbus matrix
#    Y = pd.DataFrame(n.sub_networks.obj[0].Y.todense(), buses, buses)
#    Z = pd.DataFrame(pinv(Y), buses, buses)
#    Z = Z.add(-Z.loc[slackbus])

    # -------------------------------------------------------------------------

    # difference in the first term
    Z_diff = ((Z.reindex(n.lines.bus0) - Z.reindex(n.lines.bus1).values)
              .set_index(n.lines.bus1, append=True))
    y_se = (1/(n.lines["r_pu"] + 1.j*n.lines["x_pu"])
            .set_axis(Z_diff.index, inplace=False))
    y_sh = ((n.lines["g_pu"] + 1.j*n.lines["b_pu"])
            .set_axis(Z_diff.index, inplace=False))

    # build electrical distance according to equation (7)
    A = (Z_diff.mul(y_se, axis=0)
         + Z.reindex(Z_diff.index.levels[0]).mul(y_sh, axis=0))


def marginal_welfare_contribution(n, snapshots=None, formulation='kirchhoff',
                                  return_networks=False):
    import pyomo.environ as pe
    from .opf import (extract_optimisation_results,
                      define_passive_branch_flows_with_kirchhoff)
    def fmap(f, iterable):
        # mapper for inplace functions
        for x in iterable:
            f(x)

    def profit_by_gen(n):
        price_by_generator = (n.buses_t.marginal_price
                              .reindex(columns=n.generators.bus)
                              .set_axis(n.generators.index, axis=1,
                                        inplace=False))
        revenue = price_by_generator * n.generators_t.p
        cost = n.generators_t.p.multiply(n.generators.marginal_cost, axis=1)
        return ((revenue - cost).rename_axis('profit')
                .rename_axis('generator', axis=1))

    if snapshots is None:
        snapshots = n.snapshots
    n.lopf(snapshots, solver_name='gurobi_persistent', formulation=formulation)
    m = n.model

    networks = {}
    networks['orig_model'] = n if return_networks else profit_by_gen(n)

    m.zero_flow_con = pe.ConstraintList()

    for line in n.lines.index:
#        m.solutions.load_from(n.results)
        n_temp = n.copy()
        n_temp.model = m
        n_temp.mremove('Line', [line])

        # set line flow to zero
        line_var = m.passive_branch_p['Line', line, :]
        fmap(lambda ln: m.zero_flow_con.add(ln == 0), line_var)

        fmap(n.opt.add_constraint, m.zero_flow_con.values())

        # remove cycle constraint from persistent solver
        fmap(n.opt.remove_constraint, m.cycle_constraints.values())

        # remove cycle constraint from model
        fmap(m.del_component, [c for c in dir(m) if 'cycle_constr' in c])
        # add new cycle constraint to model
        define_passive_branch_flows_with_kirchhoff(n_temp, snapshots, True)
        # add cycle constraint to persistent solver
        fmap(n.opt.add_constraint, m.cycle_constraints.values())

        # solve
        n_temp.results = n.opt.solve()
        m.solutions.load_from(n_temp.results)

        # extract results
        extract_optimisation_results(n_temp, snapshots,
                                     formulation='kirchhoff')

        if not return_networks:
            n_temp = profit_by_gen(n_temp)
        networks[line] = n_temp

        # reset model
        fmap(n.opt.remove_constraint, m.zero_flow_con.values())
        m.zero_flow_con.clear()

    return (pd.Series(networks)
            .rename_axis('removed line')
            .rename('Network'))



def flow_allocation(n, snapshots, method='Average participation', **kwargs):
    """
    Function to allocate the total network flow to buses. Available
    methods are 'Average participation' ('ap'), 'Marginal
    participation' ('mp'), 'Virtual injection pattern' ('vip'),
    'Minimal flow shares' ('mfs').



    Parameters
    ----------

    network : pypsa.Network object

    snapshots : string or pandas.DatetimeIndex
                (subset of) snapshots of the network

    per_bus : Boolean, default is False
              Whether to allocate the flow in an peer-to-peeer manner,

    method : string
        Type of the allocation method. Should be one of

            - 'Average participation'/'ap':
                Trace the active power flow from source to sink
                (or sink to source) using the principle of proportional
                sharing and calculate the partial flows on each line,
                or to each bus where the power goes to (or comes from).
            - 'Marginal participation'/'mp':
                Allocate line flows according to linear sensitvities
                of nodal power injection given by the changes in the
                power transfer distribution factors (PTDF)
            - 'Virtual injection pattern'/'vip'
                Sequentially calculate the load flow induced by
                individual power sources in the network ignoring other
                sources and scaling down sinks.
            - 'Least square color flows'/'mfs'


    Returns
    -------
    res : dict
        The returned dict consists of two values of which the first,
        'flow', represents the allocated flows within a mulitindexed
        pandas.Series with levels ['snapshots', 'bus', 'line']. The
        second object, 'cost', returns the corresponding cost derived
        from the flow allocation.
    """
#    raise error if there are no flows
    if n.lines_t.p0.shape[0] == 0:
        raise ValueError('Flows are not given by the network, '
                         'please solve the network flows first')
    n.calculate_dependent_values()

    if method in ['Average participation', 'ap']:
        method_func = average_participation
    elif method in ['Marginal Participation', 'mp']:
        method_func = marginal_participation
    elif method in ['Virtual injection pattern', 'vip']:
        method_func = virtual_injection_pattern
    elif method in ['Minimal flow shares', 'mfs']:
        method_func = minimal_flow_shares
    else:
        raise(ValueError('Method not implemented, please choose one out of'
                         "['Average participation', "
                         "'Marginal participation',"
                         "'Virtual injection pattern',"
                         "'Least square color flows']"))

    if isinstance(snapshots, str):
        snapshots = [snapshots]

    flow = pd.concat([method_func(n, sn, **kwargs) for sn in snapshots])
#    preliminary: define cost as the average usage of all lines
    cost = flow.abs().groupby(level='bus').mean()
    return {"flow": flow, "cost": cost}


def chord_diagram(allocation, lower_bound=0, groups=None, size=300,
                  save_path='/tmp/chord_diagram_pypsa'):
    """
    This function builds a chord diagram on the base of holoviews [1].
    It visualizes allocated peer-to-peer flows for all buses given in
    the data. As for compatibility with ipython shell the rendering of
    the image is passed to matplotlib however to the disfavour of
    interactivity. Note that the plot becomes only meaningful for networks
    with N > 5, because of sparse flows otherwise.


    [1] http://holoviews.org/reference/elements/bokeh/Chord.html

    Parameters
    ----------

    allocation : pandas.Series (MultiIndex)
        Series of power transmission between buses. The first index
        level ('bus/source') represents the source of the flow, the second
        level ('bus/sink') its sink.
    lower_bound : int, default is 0
        filter small power flows by a lower bound
    groups : pd.Series, default is None
        Specify the groups of your buses, which are then used for coloring.
        The series must contain values for all allocated buses.
    size : int, default is 300
        Set the size of the holoview figure
    save_path : str, default is '/tmp/chord_diagram_pypsa'
        set the saving path of your figure

    """

    import holoviews as hv
    hv.extension('matplotlib')
    from IPython.display import Image

    if len(allocation.index.levels) == 3:
        allocation = allocation[allocation.index.levels[0][0]]

    allocated_buses = allocation.index.levels[0] \
                      .append(allocation.index.levels[1]).unique()
    bus_map = pd.Series(range(len(allocated_buses)), index=allocated_buses)

    links = allocation.to_frame('value').reset_index()\
        .replace({'bus/source': bus_map, 'bus/sink': bus_map})\
        .sort_values('bus/source').reset_index(drop=True) \
        [lambda df: df.value >= lower_bound]

    nodes = pd.DataFrame({'bus': bus_map.index})
    if groups is None:
        cindex = 'index'
        ecindex = 'bus/source'
    else:
        groups = groups.rename(index=bus_map)
        nodes = nodes.assign(groups=groups)
        links = links.assign(groups=links['bus/source']
                             .map(groups))
        cindex = 'groups'
        ecindex = 'groups'

    nodes = hv.Dataset(nodes, 'index')
    diagram = hv.Chord((links, nodes))
    diagram = diagram.opts(style={'cmap': 'Category20',
                                  'edge_cmap': 'Category20'},
                           plot={'label_index': 'bus',
                                 'color_index': cindex,
                                 'edge_color_index': ecindex
                                 })
    renderer = hv.renderer('matplotlib').instance(fig='png', holomap='gif',
                                                  size=size, dpi=300)
    renderer.save(diagram, 'example_I')
    return Image(filename='example_I.png', width=800, height=800)




