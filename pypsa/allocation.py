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
    
def PTDF(network, slacked=False):
    if slacked:
        network.determine_network_topology()
        calculate_PTDF(network.sub_networks.obj[0])
        return pd.DataFrame(network.sub_networks.obj[0].PTDF
                      , columns=network.buses.index, index=network.lines.index)
    else:
        K = pd.DataFrame(network.incidence_matrix(busorder=network.buses.index).todense())
        Omega = pd.DataFrame(np.diagflat(1/network.lines.x_pu.values))
        return pd.DataFrame(Omega.dot(K.T).dot(np.linalg.pinv(K.dot(Omega).dot(K.T))
            ).values, columns=network.buses.index, index=network.lines.index)


#%% 

def average_participation(network, snapshot, per_bus=False, normalized=False, 
                          downstream=True):
#   principally follow Hoersch, Jonas; "Flow tracing as a tool set for the 
#   analysis of networked large-scale renewable electricity systems"
#   and use matrix notation to derive Q.

    buses = network.buses.index
    lines = network.lines.index
    links = network.links.index
    f_in = (network.lines_t.p0.loc[[snapshot]].T
             .append(network.links_t.p0.loc[ [snapshot]].T))
    f_out =  (- network.lines_t.p1.loc[[snapshot]].T
             .append(network.links_t.p1.loc[[snapshot]].T))
    p = network.buses_t.p.loc[[snapshot]].T
    K = pd.DataFrame(network.incidence_matrix(branch_components={'Line', 'Link'}, 
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
            return Q
        else:
#            equal to weighting and stacking (from above) with R:
            return Q.swaplevel(0).sort_index()

#    create artificial injection patterns following the flow trace    
    if downstream:
        T = Q.copy()
        weight = (p_in + p_out)[snapshot]
        ref_b = p_in[snapshot] == 0
        tag = '/source'
    else:
        T = R.copy()
        weight = F_in
        ref_b = p_out[snapshot] == 0
        tag = '/sink'
        
    T = T.mul((weight), axis=0)
    T = (pd.DataFrame(np.diagflat(np.diag(T)), index=T.index, columns=T.columns) \
         - T[ref_b].reindex_like(T).fillna(0))
    T = PTDF(network).dot(T).round(10)
    if normalized:
        T.div(f_in[snapshot], axis=0)
    T = T.rename_axis('line').rename_axis('bus' + tag , axis=1).T
    T = T.stack()[lambda ds:ds!=0]
    return pd.concat([T], keys=[snapshot], names=['snapshots'])


def marginal_participation(network, snapshot, Q=0.5, normalized=False):
    H = PTDF(network)
    p = network.buses_t.p.loc[snapshot]
    p_plus = p.clip_lower(0)
    f = network.lines_t.p0.loc[snapshot]
#    f_plus = f.clip_lower(0)
    f_plus = H.dot(p_plus)
    k_plus = (Q*f - f_plus)/p_plus.sum()
    if normalized:
        c = H.add(k_plus, axis=0).mul(p, axis=1).div(f, axis=0).fillna(0).T
    else:
        c = H.add(k_plus, axis=0).mul(p, axis=1).fillna(0).T
    c = c.rename_axis('bus').rename_axis("line", axis=1)
    c = c.stack().round(8)[lambda ds:ds!=0]
    return pd.concat([c], keys=[snapshot], names=['snapshots'])


def virtual_injection_pattern(network, snapshot, normalized=False):
    H = PTDF(network)
    p = network.buses_t.p.loc[snapshot]
    p_plus = p.clip_lower(0)
    p_minus = p.clip_upper(0)
    f = network.lines_t.p0.loc[snapshot]
    indiag = diag(p_plus)
    offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T).div(p_plus.sum())
                .pipe(lambda df: df - np.diagflat(np.diag(df))) )
    vip = indiag + offdiag
#    normalized colorvectors
    if normalized:
        c = H.dot(vip).div(f, axis=0).fillna(0).T
    else:
        c = H.dot(vip).fillna(0).T
    c = c.rename_axis('bus').rename_axis("line", axis=1)
    c = c.stack().round(8)[lambda ds:ds!=0]
    return pd.concat([c], keys=[snapshot], names=['snapshots'])


def minimized_flow_shares(network, snapshot, **kwargs):
    
    from scipy.optimize import minimize
    H = PTDF(network)
    #    f = network.lines_t.p0.loc[snapshot]
    p = network.buses_t.p.loc[snapshot]
    indiag = diag(p*p)
    offdiag = p.to_frame().dot(p.to_frame().T).clip_upper(0)
    pp = indiag + offdiag
    N = len(network.buses)
    
    def minimization(df):
        return ((H.dot(df.reshape(N,N)))**2).sum().sum()
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
    sol = pd.DataFrame(sol.x.reshape(N,N), columns=network.buses.index, index=network.buses.index).round(10)    
    c = H.dot(sol).round(2)
    return c



def flow_allocation(network, snapshots, method='Average participation'):
    """
    Function to allocate the total network flow to buses. Available methods are
    'Average participation', 'Marginal participation', 'Virtual injection pattern',
    'Least square color flows'. 
    
    
    
    Parameters
    ----------
    
    network : pypsa.Network object 
    
    snapshots : string or pandas.DatetimeIndex
                (subset of) snapshots of the network
                
    method : string
        Type of the allocation method. Should be one of
                
            - 'Average participation': 
            - 'Marginal participation':
            - 'Virtual injection pattern':
            - 'Least square color flows': 
                
    Returns
    -------
#    not yet there
    res : AllocatedFlow
        The allocated flow represented in an 'AllocatedFlow' object.
        The attribute 'flows' returns the allocated flows in form of a pd.Series 
        with MultiIndex with levels ['snapshots', 'bus', 'line'] 
    """
#    raise error if there are no flows    
    if network.lines_t.p0.shape[0] == 0:
        raise ValueError('Flows are not given by the network, please solve the network flows first')
    
    if method=='Average participation':
        method_func = average_participation
    elif method=='Marginal Participation':
        method_func = marginal_participation
    elif method=='Virtual injection pattern':
        method_func = virtual_injection_pattern
    elif method=='Least square color flows':
        method_func = minimization_method
    else:
         raise(ValueError(""""Method not implemented, please choose one out of 
                          ['Average participation', 'Marginal participation',
                           'Virtual injection pattern','Least square color flows']"""))

    if isinstance(snapshots, str):
        snapshots = [snapshots]
    
    flow = pd.concat([method_func(network, sn) for sn in snapshots])
#    preliminary: define cost as the average usage of all lines 
    cost = flow.abs().groupby(level='bus').mean()
    return {"flow":flow, "cost":cost}
    
          

#%% testings
    

#def flow_distribution_factors(network, snapshot):
#    F = diag(network.lines_t.p0.loc[snapshot])
#    K = pd.DataFrame(network.incidence_matrix(busorder=network.buses.index).todense(),
#                        columns=network.lines.index, index=network.buses.index)
#    Z = pd.DataFrame(F.dot(K.T).dot(np.linalg.pinv(K.dot(F).dot(K.T))
#            ).values, columns=network.buses.index, index=network.lines.index)
#    P = diag(network.buses_t.p.loc[snapshot]).clip_lower(0)
#    P_minus = network.buses_t.p.loc[snapshot].clip_upper(0)
#    P.apply(lambda df: df - P_minus/P_minus.sum()*df.sum())
#    return Z.dot(P)    


    

