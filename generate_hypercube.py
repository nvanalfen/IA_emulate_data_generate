import numpy as np
from smt.sampling_methods import LHS

param_ranges = {
    "central_alignment_strength":[-1.,1.],
    "satellite_alignment_strength":[-1.,1.],
    "logMmin":[12.0,13.0],
    "sigma_logM":[0.2,0.3],
    "logM0":[12.,13.],
    "logM1":[13.,14.],
    "alpha":[0.5,1.5]
}

# Make the hypercube
limits = np.array( [ param_ranges[key] for key in param_ranges ] )
sampler = LHS(xlimits=limits)

N = 10000
hypercube = sampler(N)

np.savez("hypercube.npz", keys=np.array([key for key in param_ranges]), hypercube=hypercube)