import numpy as np
from smt.sampling_methods import LHS

# The ranges for four of the five occupation parameters are taken from the zheng07_components.py file in halotools
# which come from the table in Zheng et al. 2007 (fllored or ceilinged to one decimal place)
# The exception being alpha, whose range in that table went from about 0.8 to 1.15, but I chose to extend
param_ranges = {
    "central_alignment_strength":[-1.,1.],
    "satellite_alignment_strength":[-1.,1.],
    "logMmin":[11.0,15.0],
    "sigma_logM":[0.2,0.8],
    "logM0":[11.,14.],
    "logM1":[12.,15.],
    "alpha":[0.5,1.5]
}

# Make the hypercube
limits = np.array( [ param_ranges[key] for key in param_ranges ] )
sampler = LHS(xlimits=limits)

N = 100000
hypercube = sampler(N)

np.savez("bolplanck_hypercube.npz", keys=np.array([key for key in param_ranges]), hypercube=hypercube)