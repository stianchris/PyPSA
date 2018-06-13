#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:14:49 2018

@author: fabian
"""

#This side-package is created for use as flow and cost allocation. 

import pandas as pd
import pypsa as ps
import numpy as np
import xarray as xr
import logging
logger = logging.getLogger(__name__)
from .pf import calculate_PTDF
from numpy.linalg import inv, pinv



#%% utility functions
def diag(series):
    return pd.DataFrame(np.diagflat(series.values), 
                        index=series.index, columns=series.index)

def incidence_matrix(n):
    buses = n.buses.index
    lines = n.lines.index
    links = n.links.index
    return pd.DataFrame(n.incidence_matrix(branch_components={'Line', 'Link'}, 
                                                 busorder=buses).todense(), 
                         index=buses, columns=lines.append(links))
    
def PTDF(n, slacked=False):
    if slacked:
        n.determine_n_topology()
        calculate_PTDF(n.sub_ns.obj[0])
        return pd.DataFrame(n.sub_ns.obj[0].PTDF
                      , columns=n.buses.index, index=n.lines.index)
    else:
        K = pd.DataFrame(n.incidence_matrix(busorder=n.buses.index).todense())
        Omega = pd.DataFrame(np.diagflat(1/n.lines.x_pu.values))
        return pd.DataFrame(Omega.dot(K.T).dot(np.linalg.pinv(K.dot(Omega).dot(K.T))
            ).values, columns=n.buses.index, index=n.lines.index)


#%% 

def average_participation(n, snapshot, per_bus=False, normalized=False, 
                          downstream=True):
#   principally follow Hoersch, Jonas; "Flow tracing as a tool set for the 
#   analysis of ned large-scale renewable electricity systems"
#   and use matrix notation to derive the downstream allocation Q.
    """
    Allocate the network flow in according to the method 'Average participation' or 'Flow tracing'
    firstly presented in [1,2]. The algorithm itself is derived from [3]. The general idea is to 
    follow active power flow from source to sink (or sink to source) using the principle of 
    proportional sharing and calculate the partial flows on each line, or to each bus where the
    power goes to (or comes from). 
    
    This method provdes two general options:
        Downstream: 
            The flow of each nodal power injection is traced through the network and decomposed 
            the to set of lines/buses on which is flows on/to.  
        Upstream:
            The flow of each nodal power demand is traced (in reverse direction) through the network
            and decomposed to the set of lines/buses where it comes from. 
    
    [1] J. Bialek, “Tracing the flow of electricity,” 
        IEE Proceedings - Generation, Transmission and Distribution, vol. 143, no. 4, p. 313, 1996.
    [2] D. Kirschen, R. Allan, G. Strbac, Contributions of individual generators
        to loads and flows, Power Systems, IEEE Transactions on 12 (1) (1997)
        52–60. doi:10.1109/59.574923.
    [3] J. Hörsch, M. Schäfer, S. Becker, S. Schramm, and M. Greiner, “Flow trac-
        ing as a tool set for the analysis of networked large-scale renewable electric-
        ity systems,” International Journal of Electrical Power & Energy Systems,
        vol. 96, pp. 390–397, Mar. 2018.



    Parameters
    ----------
    network : pypsa.Network() object with calculated flow data

    snapshot : str
        Specify snapshot which should be investigated. Must be in network.snapshots.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow 
    downstream : Boolean, default True
        Whether to use downstream or upstream method. 

    """
    
    
    
    buses = n.buses.index
    lines = n.lines.index
    links = n.links.index
    f_in = (n.lines_t.p0.loc[[snapshot]].T
             .append(n.links_t.p0.loc[ [snapshot]].T))
    f_out =  (- n.lines_t.p1.loc[[snapshot]].T
             .append(n.links_t.p1.loc[[snapshot]].T))
    p = n.buses_t.p.loc[[snapshot]].T
    K = pd.DataFrame(n.incidence_matrix(branch_components={'Line', 'Link'}, 
                                              busorder=buses).todense(), 
                     index=buses, columns=lines.append(links))
#    set incidence matrix in direction of flow
    K = K.mul(np.sign(f_in[snapshot]))
    f_in = f_in.abs()    
    f_out = f_out.abs()

    n_in =  K.clip_upper(0).dot(f_in).abs() #network inflow
    n_out = K.clip_lower(0).dot(f_out) #network outflow
    p_in = p.clip_lower(0) #nodal inflow
    p_out = p.clip_upper(0).abs() #nodal outflow
    
    # flows from bus (index) to bus (columns):
    F = (K.dot(diag(f_in)).clip_lower(0).dot(K.T).clip_upper(0).abs()) 

    F_in = (p_in + n_in)[snapshot] 
#   F_in = F_out = (p_out + n_out)[snapshot]

#   upstream
    Q = pd.DataFrame(inv(diag(F_in) - F.T).dot(diag(p_in)),
                     index = buses, columns = buses)
#   downstream
    R = pd.DataFrame(inv(diag(F_in) - F).dot(diag(p_out)),
                     index = buses, columns = buses)
#    equation (12)
#    (Q.T.dot(diag(p_out)).T == R.T.dot(diag(p_in)) ).all().all()    
    
    if per_bus and normalized:
        return Q
    if per_bus:
        Q = (Q.mul(p_out[snapshot], axis=0)
                .rename_axis('bus/sink').rename_axis('bus/source', axis=1)
                .stack()[lambda ds:ds!=0])
        if downstream:
            T = Q
        else:
#            equal to weighting and stacking (from above) with R:
            T = Q.swaplevel(0).sort_index()

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
        T = (pd.DataFrame(np.diagflat(np.diag(T)), index=T.index, columns=T.columns) \
             - T[ref_b].reindex_like(T).fillna(0))
        T = PTDF(n).dot(T).round(10)
        if normalized:
            T.div(f_in[snapshot], axis=0)
        T = T.rename_axis('line').rename_axis('bus' + tag , axis=1).T
        T = T.stack()[lambda ds:ds!=0]
    return pd.concat([T], keys=[snapshot], names=['snapshots'])

    
def marginal_participation(n, snapshot, q=0.5, normalized=False,
                           per_bus=False):
    '''
    Allocate line flows according to linear sensitvities of nodal power injection given by the
    changes in the power transfer distribution factors (PTDF)[1-3]. As the method is based on the DC-
    approximation, it works on subnetworks only as link flows are not taken into account. 
    
    
    [1] F. J. Rubio-Oderiz, I. J. Perez-Arriaga, Marginal pricing of transmission
        services: a comparative analysis of network cost allocation methods, IEEE
        Transactions on Power Systems 15 (1) (2000) 448–454. doi:10.1109/59.
        852158.
    [2] M. Schäfer, B. Tranberg, S. Hempel, S. Schramm, M. Greiner, Decomposi-
        tions of injection patterns for nodal flow allocation in renewable electricity
        networks, The European Physical Journal B 90 (8) (2017) 144.
    [3] T. Brown, “Transmission network loading in Europe with high shares of renewables,” 
        IET Renewable Power Generation, vol. 9, no. 1, pp. 57–65, Jan. 2015.


    Parameters
    ----------
    network : pypsa.Network() object with calculated flow data

    snapshot : str
        Specify snapshot which should be investigated. Must be in network.snapshots.
    q : float, default 0.5 
        split between net producers and net consumers. If q is zero, only the impact of net
        load is taken into account. If q is one, only net generators are taken into account.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines if False.
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
             .stack().round(8)[lambda ds:ds!=0])
    else:    
        Q = (Q.rename_axis('bus').rename_axis("line", axis=1)
             .stack().round(8)[lambda ds:ds!=0])
    return pd.concat([Q], keys=[snapshot], names=['snapshots'])


def virtual_injection_pattern(n, snapshot, normalized=False, per_bus=False,
                              downstream=True):
    H = PTDF(n)
    p = n.buses_t.p.loc[snapshot]
    p_plus = p.clip_lower(0)
    p_minus = p.clip_upper(0)
    f = n.lines_t.p0.loc[snapshot]
    if downstream:
        indiag = diag(p_plus)
        offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T).div(p_plus.sum())
                    .pipe(lambda df: df - np.diagflat(np.diag(df))) )
    else:
        indiag = diag(p_minus)
        offdiag = (p_plus.to_frame().dot(p_minus.to_frame().T).div(p_minus.sum())
                    .pipe(lambda df: df - np.diagflat(np.diag(df))) )
    vip = indiag + offdiag
    if per_bus:
        Q = (vip[indiag.sum()==0].T
                .rename_axis('bus/sink', axis=int(downstream))
                .rename_axis('bus/source', axis=int(not downstream))
                .stack()[lambda ds:ds!=0]).abs() 
#        switch to counter stream by Q.swaplevel(0).sort_index()
    else:
        Q = H.dot(vip).round(10).T
        if normalized:
    #       normalized colorvectors
            Q /= f
        Q = (Q.rename_axis('bus').rename_axis("line", axis=1)
             .stack().round(8)[lambda ds:ds!=0])
    return pd.concat([Q], keys=[snapshot], names=['snapshots'])


def minimal_flow_shares(n, snapshot, **kwargs):
    
    from scipy.optimize import minimize
    H = PTDF(n)
    #    f = n.lines_t.p0.loc[snapshot]
    p = n.buses_t.p.loc[snapshot]
    indiag = diag(p*p)
    offdiag = p.to_frame().dot(p.to_frame().T).clip_upper(0)
    pp = indiag + offdiag
    N = len(n.buses)
    
    def minimization(df):
        return -((H.dot(df.reshape(N,N)))**2).sum().sum()
    constr = [
            #   nodal balancing
            {'type':'eq', 'fun':(lambda df: df.reshape(N,N).sum(0) ) },
            #    total injection of colors
            {'type':'eq', 'fun':(lambda df: df.reshape(N,N).sum(1)-p) },
            #   flow conservation
    #        {'type':'eq', 'fun':(lambda df: H.dot(df.reshape(6,6)).sum(1) - f) }
                ]    
    
    #   sources-sinks-fixation
    bounds = pp.unstack()
    bounds = pd.concat([bounds.clip_upper(0), bounds.clip_lower(0)], axis=1)
    bounds[bounds!=0] = None
    
    sol = minimize(minimization, pp, constraints=constr, bounds=bounds.values, 
                   options={'maxiter':400}, tol=1e-12, **kwargs)
    print sol.message
    sol = pd.DataFrame(sol.x.reshape(N,N), columns=n.buses.index, index=n.buses.index).round(10)    
    c = H.dot(sol).round(2)
    return c



def flow_allocation(n, snapshots, method='Average participation', **kwargs):
    """
    Function to allocate the total network flow to buses. Available methods are
    'Average participation', 'Marginal participation', 'Virtual injection pattern',
    'Minimal flow shares'. 
    
    
    
    Parameters
    ----------
    
    network : pypsa.Network object 
    
    snapshots : string or pandas.DatetimeIndex
                (subset of) snapshots of the network
                
    per_bus : Boolean, default is False
              Whether to allocate the flow in an peer-to-peeer manner, 
                
    method : string
        Type of the allocation method. Should be one of
                
            - 'Average participation': 
            - 'Marginal participation':
            - 'Virtual injection pattern':
            - 'Least square color flows': 
                
    Returns
    -------
    res : dict
        The returned dict consists of two values of which the first, 'flow', 
        represents the allocated flows within a mulitindexed pandas.Series 
        with levels ['snapshots', 'bus', 'line']. The second object, 'cost', 
        returns the corresponding cost derived from the flow allocation. 
    """
#    raise error if there are no flows    
    if n.lines_t.p0.shape[0] == 0:
        raise ValueError('Flows are not given by the network, please solve the network flows first')
    
    if method=='Average participation':
        method_func = average_participation
    elif method=='Marginal Participation':
        method_func = marginal_participation
    elif method=='Virtual injection pattern':
        method_func = virtual_injection_pattern
    elif method=='Minimal flow shares':
        method_func = minimal_flow_shares
    else:
         raise(ValueError(""""Method not implemented, please choose one out of 
                          ['Average participation', 'Marginal participation',
                           'Virtual injection pattern','Least square color flows']"""))

    if isinstance(snapshots, str):
        snapshots = [snapshots]
    
    flow = pd.concat([method_func(n, sn, **kwargs) for sn in snapshots])
#    preliminary: define cost as the average usage of all lines 
    cost = flow.abs().groupby(level='bus').mean()
    return {"flow":flow, "cost":cost}

