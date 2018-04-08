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
from numpy.linalg import inv



#%% utility functions
def to_diag(series):
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

def average_participation(network, snapshot, normalized=False, return_exports=True):
    buses = network.buses.index
    lines = network.lines.index
#   use matrix formulation for flow allocation
    K = pd.DataFrame(network.incidence_matrix(branch_components={'Line', 'Link'}, busorder=buses).todense(), 
                     index=buses, columns=lines)
    F = to_diag(network.lines_t.p0.loc[snapshot])
    P_plus = to_diag(network.buses_t.p.loc[snapshot])
    exported_source_flows = K.dot(F).clip_lower(0).dot(np.sign(F)).dot(K.T) \
                        .clip_upper(0).pipe(lambda df:df/df.abs().sum().replace(0,1))
#   define colorflow matrix
    C = P_plus.dot( 
            pd.DataFrame(inv(np.eye(len(buses)) - exported_source_flows) ,
                         index=buses, columns=buses) )
#   normalize such that each column of the Colormatrix sums up to one
    if normalized:
        C = C.pipe(lambda df:df/df.sum())
    if return_exports:
    #   define colorized exports on each line
        exports = C.dot(K).clip_lower(0)
    #   normalize the exports such that they are equal to the line flow
        exports *= F.sum()/exports.sum().replace(0,1)
        #transform to multi-index to simplify export to xarray         
        exports = exports.rename_axis('bus').rename_axis("line", axis=1)
        exports = exports.stack()[lambda ds:ds!=0]
        return pd.concat([exports], keys=[snapshot], names=['snapshots'])

#   likewise define imports, not correct yet
    imports = C.dot(K).clip_upper(0)
    imports /= imports.sum()*F.sum()    
    return imports.stack()[lambda ds:ds!=0]




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
    diag = to_diag(p_plus)
    offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T).div(p_plus.sum())
                .pipe(lambda df: df - np.diagflat(np.diag(df))) )
    vip = diag + offdiag
#    normalized colorvectors
    if normalized:
        c = H.dot(vip).div(f, axis=0).fillna(0).T
    else:
        c = H.dot(vip).fillna(0).T
    c = c.rename_axis('bus').rename_axis("line", axis=1)
    c = c.stack().round(8)[lambda ds:ds!=0]
    return pd.concat([c], keys=[snapshot], names=['snapshots'])


def minimization_method(network, snapshot, **kwargs):
    
    from scipy.optimize import minimize
    H = PTDF(network)
    #    f = network.lines_t.p0.loc[snapshot]
    p = network.buses_t.p.loc[snapshot]
    diag = to_diag(p*p)
    offdiag = p.to_frame().dot(p.to_frame().T).clip_upper(0)
    pp = diag + offdiag
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


def flow_allocation(network, snapshots, method='average_participation'):
    """
    Function to allocate the total network flow to buses. Available methods are
    'Average participation', 'Marginal Participation', 'Virtual Injection Pattern',
    ...
    
    
    
    Parameters
    ----------
    
    network : pypsa.Network object 
    
    snapshots : string or pandas.DatetimeIndex
                (subset of) snapshots of the network
                
    method : string
        Type of the allocation method. Should be one of
                
            - 'Average participation': 
            - 'Marginal participation':
            - 'Virtual Injection Pattern':
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
    
    if method=='average_participation':
        method_func = average_participation
#    elif 
    
#    initialize basic quanitties
#    flow = xr.DataArray(network.lines_t.p0)
#    flow.attrs['units'] = 'MW'
    if isinstance(str, snapshots):
        snapshots = [snapshots]
    
    flow = pd.concat([method_func(network, sn) for sn in snapshots])
#    preliminary: define cost as the average usage of all lines 
    cost = flow.abs().groupby(level='bus').mean()
    return flow, cost
    
          
