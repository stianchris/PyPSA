## Minimal 3-node example of PyPSA power flow
#
#Available as a Jupyter notebook at <http://www.pypsa.org/examples/minimal_example_pf.ipynb>.


# make the code as Python 3 compatible as possible
from __future__ import print_function, division
import pypsa

import numpy as np

network = pypsa.Network()

#add three buses
n_buses = 3

for i in range(n_buses):
    network.add("Bus","My bus {}".format(i),
                v_nom=20.)

print(network.buses)

#add three lines in a ring
for i in range(n_buses):
    network.add("Line","My line {}".format(i),
                bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1)%n_buses),
                x=0.1,
                r=0.01)

print(network.lines)

#add a generator at bus 0
network.add("Generator","My gen",
            bus="My bus 0",
            p_set=100,
            control="PQ")

#network.add("ShuntImpedance","My shunt",
#            bus="My bus 1",
#            g=0.6,
#            b=0.3,
#            v_nom=10)
#

print(network.generators)

print(network.generators.p_set)

#add a load at bus 1
network.add("Load","My load",
            bus="My bus 1",
            p_set=100)

print(network.loads)

print(network.loads.p_set)

network.loads.q_set = 100.

#Do a Newton-Raphson power flow
network.pf()

print(network.lines_t.p0)

print(network.buses_t.v_ang*180/np.pi)

print(network.buses_t.v_mag_pu)


#%%
import pandas as pd
from pypsa.allocation import incidence_matrix, diag, pinv
from numpy import conj
from numpy.testing import assert_array_almost_equal as as_equal
upper = lambda df : df.clip_lower(0)
lower = lambda df : df.clip_upper(0)


n = network
n.calculate_dependent_values()
snapshot = n.snapshots[0]
slackbus = n.buses[(n.buses_t.v_ang == 0).all()].index[0]


# linearised method, start from linearised admittance matrix
y = pd.concat([1/(n.lines.r_pu + 1.j * n.lines.x_pu)], keys=['Line'])

K = incidence_matrix(n, branch_components=['Line'])
Y = K @ diag(y) @ K.T  # Ybus matrix

v = (n.buses_t.v_mag_pu * np.exp( 1.j * n.buses_t.v_ang)).T
i = Y @ v
s = n.buses_t.p + 1.j * n.buses_t.q
as_equal(v * np.conj(i), s.T)


i_ = diag(y) @ K.T @ v



s0_ = (n.lines_t.p0 + 1.j * n.lines_t.q0).T
s1_ = (n.lines_t.p1 + 1.j * n.lines_t.q1).T

as_equal(s0_['now'], n.lines.bus0.map(v['now']) * np.conj(i_.loc['Line']['now']))
as_equal(s1_['now'], -n.lines.bus1.map(v['now']) * np.conj(i_.loc['Line']['now']))
