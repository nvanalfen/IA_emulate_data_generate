from __future__ import print_function, division
import numpy as np
import sys
import os
import itertools

import time

from halotools.sim_manager import CachedHaloCatalog

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models.ia_models.ia_model_components import CentralAlignment, RadialSatelliteAlignment
from halotools.empirical_models.ia_models.ia_strength_models import RadialSatelliteAlignmentStrength
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens, Zheng07Sats, SubhaloPhaseSpace
from halotools.mock_observables import tpcf
from halotools.mock_observables.ia_correlations import ee_3d, ed_3d

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

############################################################################################################################
##### FUNCTIONS ############################################################################################################
############################################################################################################################

def read_variables():
    # TODO: Grab file and job number from sys if relevant
    param_file = "generate_constant.npy"
    job = None
    max_jobs = None
    if len(sys.argv) > 1:
        param_file = sys.argv[1]
        if len(sys.argv) > 2:
            job = int(sys.argv[2])
            if len(sys.argv) > 3:
                max_jobs = int(sys.argv[3])
            else:
                max_jobs = job

    # TODO: Set/Retrieve lists of alignment parameters to loop through
    parameters = np.load(param_file, allow_pickle=True).item()

    parameters["job"] = job
    parameters["max_jobs"] = max_jobs

    return parameters

# Eliminate halos with 0 for halo_axisA_x(,y,z)
def mask_bad_halocat(halocat):
    bad_mask = (halocat.halo_table["halo_axisA_x"] == 0) & (halocat.halo_table["halo_axisA_y"] == 0) & (halocat.halo_table["halo_axisA_z"] == 0)
    bad_mask = bad_mask ^ np.ones(len(bad_mask), dtype=bool)
    halocat._halo_table = halocat.halo_table[ bad_mask ]

def build_model_instance(cen_strength, sat_params, sat_bins, halocat, constant=True, seed=None):

    sat_alignment_strength = 1

    if constant:
        sat_alignment_strength = sat_params
    else:
        sat_a, sat_gamma = sat_params

    cens_occ_model = Zheng07Cens()
    cens_prof_model = TrivialPhaseSpace()
    cens_orientation = CentralAlignment(central_alignment_strenth=cen_strength)

    sats_occ_model = Zheng07Sats()
    prof_args = ("satellites", sat_bins)
    sats_prof_model = SubhaloPhaseSpace(*prof_args)

    sats_orientation = RadialSatelliteAlignment(satellite_alignment_strength=sat_alignment_strength, halocat=halocat)
    if not constant:
        sats_strength = RadialSatelliteAlignmentStrength(satellite_alignment_a=sat_a, satellite_alignment_gamma=sat_gamma)
        Lbox = halocat.Lbox
        sats_strength.inherit_halocat_properties(Lbox=Lbox)
    
    if constant:
        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                        centrals_profile = cens_prof_model,
                                        satellites_occupation = sats_occ_model,
                                        satellites_profile = sats_prof_model,
                                        #satellites_radial_alignment_strength = sats_strength,
                                        centrals_orientation = cens_orientation,
                                        satellites_orientation = sats_orientation,
                                        model_feature_calling_sequence = (
                                        'centrals_occupation',
                                        'centrals_profile',
                                        'satellites_occupation',
                                        'satellites_profile',
                                        #'satellites_radial_alignment_strength',
                                        'centrals_orientation',
                                        'satellites_orientation')
                                        )
    else:
        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                        centrals_profile = cens_prof_model,
                                        satellites_occupation = sats_occ_model,
                                        satellites_profile = sats_prof_model,
                                        satellites_radial_alignment_strength = sats_strength,
                                        centrals_orientation = cens_orientation,
                                        satellites_orientation = sats_orientation,
                                        model_feature_calling_sequence = (
                                        'centrals_occupation',
                                        'centrals_profile',
                                        'satellites_occupation',
                                        'satellites_profile',
                                        'satellites_radial_alignment_strength',
                                        'centrals_orientation',
                                        'satellites_orientation')
                                        )

    model_instance.populate_mock(halocat,seed=seed)
    
    return model_instance

def correlate(row):
    func, args, kwargs = row
    return func(*args, **kwargs)

def generate_correlations_parallel(model, rbins, halocat):
    gal_table = model.mock.galaxy_table
    cen_cut = gal_table[ gal_table["gal_type"] == "centrals" ]
    sat_cut = gal_table[ gal_table["gal_type"] == "satellites" ]

    coords = np.array( [ gal_table["x"], gal_table["y"], gal_table["z"] ] ).T
    orientations = np.array( [ gal_table["galaxy_axisA_x"], gal_table["galaxy_axisA_y"], gal_table["galaxy_axisA_z"] ] ).T
    cen_coords = np.array( [ cen_cut["x"], cen_cut["y"], cen_cut["z"] ] ).T
    cen_orientations = np.array( [ cen_cut["galaxy_axisA_x"], cen_cut["galaxy_axisA_y"], cen_cut["galaxy_axisA_z"] ] ).T
    sat_coords = np.array( [ sat_cut["x"], sat_cut["y"], sat_cut["z"] ] ).T
    sat_orientations = np.array( [ sat_cut["galaxy_axisA_x"], sat_cut["galaxy_axisA_y"], sat_cut["galaxy_axisA_z"] ] ).T
    
    func_params = [
            ( tpcf, (coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (cen_coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (cen_coords, cen_orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (cen_coords, cen_orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (sat_coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (sat_coords, sat_orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (sat_coords, sat_orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (coords, orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (coords, orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (cen_coords, rbins, coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (cen_coords, cen_orientations, coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (cen_coords, cen_orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (coords, orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (coords, orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (sat_coords, rbins, coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (sat_coords, sat_orientations, coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (sat_coords, sat_orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (cen_coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (cen_coords, cen_orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (cen_coords, cen_orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (sat_coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (sat_coords, sat_orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (sat_coords, sat_orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} )
            ]
    
    with mp.Pool() as pool:
        results = pool.map(correlate, func_params)
    
    return results

    # xi = tpcf(coords, rbins, coords, period=halocat.Lbox)
    # omega = ed_3d(coords, orientations, coords, rbins, period=halocat.Lbox)
    # eta = ee_3d(coords, orientations, coords, orientations, rbins, period=halocat.Lbox)
    
    # return [xi, omega, eta]

def generate_correlations_series(model, rbins, halocat):
    gal_table = model.mock.galaxy_table
    cen_cut = gal_table[ gal_table["gal_type"] == "centrals" ]
    sat_cut = gal_table[ gal_table["gal_type"] == "satellites" ]

    coords = np.array( [ gal_table["x"], gal_table["y"], gal_table["z"] ] ).T
    orientations = np.array( [ gal_table["galaxy_axisA_x"], gal_table["galaxy_axisA_y"], gal_table["galaxy_axisA_z"] ] ).T
    cen_coords = np.array( [ cen_cut["x"], cen_cut["y"], cen_cut["z"] ] ).T
    cen_orientations = np.array( [ cen_cut["galaxy_axisA_x"], cen_cut["galaxy_axisA_y"], cen_cut["galaxy_axisA_z"] ] ).T
    sat_coords = np.array( [ sat_cut["x"], sat_cut["y"], sat_cut["z"] ] ).T
    sat_orientations = np.array( [ sat_cut["galaxy_axisA_x"], sat_cut["galaxy_axisA_y"], sat_cut["galaxy_axisA_z"] ] ).T
    
    func_params = [
            ( tpcf, (coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (cen_coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            ( ed_3d, (cen_coords, cen_orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (cen_coords, cen_orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (sat_coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            ( ed_3d, (sat_coords, sat_orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (sat_coords, sat_orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (cen_coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (cen_coords, cen_orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (cen_coords, cen_orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (sat_coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (sat_coords, sat_orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (sat_coords, sat_orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (cen_coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            ( ed_3d, (cen_coords, cen_orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (cen_coords, cen_orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            ( tpcf, (sat_coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            ( ed_3d, (sat_coords, sat_orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (sat_coords, sat_orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} )
            ]
    
    results = []

    for func, args, kwargs in func_params:
        results.append(func(*args, **kwargs))
    
    return results

def generate_training_data(model, rbins, cen_strengths, sat_params, halocat, inner_runs=100, constant=True):
    inputs = []
    outputs = []

    for cen_mu in cen_strengths:
        for sat_param in sat_params:
            # if constant, then sat_param will be sat_mu
            # otherwise, it will me sat_a, sat_gamma
            for run in range(inner_runs):
                input_row = [cen_mu]
                
                # Repopulate model
                model.param_dict["central_alignment_strength"] = cen_mu
                if constant:
                    model.param_dict["satellite_alignment_strength"] = sat_param
                    input_row.append( sat_param )
                else:
                    sat_a, sat_gamma = sat_param
                    model.param_dict["satellite_alignment_a"] = sat_a
                    model.param_dict["satellite_alignment_gamma"] = sat_gamma
                    input_row.append( sat_a )
                    input_row.append( sat_gamma )
                model.mock.populate()

                # Add other properties to input row
                # TODO: Add other properties to input row

                # Append input and output row
                inputs.append( input_row )
                outputs.append( generate_correlations_parallel(model, rbins, halocat) )

    return np.array(inputs), np.array(outputs)

def generate_training_data_complex(model, rbins, model_param_dict, halocat, inner_runs=100, job=None, max_jobs=None):
    inputs = []
    outputs = []

    keys, values = permute_params(model_param_dict)

    start = 0
    end = len(values)
    if not job is None:
        span = int( np.ceil(len(values)/max_jobs) )
        start = (job-1)*span
        end = start+span

    ind = 0
    for value in values[start:end]:
        print(f"Value set {ind}")
        ind += 1

        # Adjust model parameters
        for i in range(len(keys)):
                model.param_dict[keys[i]] = value[i]

        for run in range(inner_runs):
            input_row = list(value)
            
            # Repopulate model
            model.mock.populate()

            # Append input and output row
            inputs.append( input_row )
            outputs.append( generate_correlations_parallel(model, rbins, halocat) )

    return keys, np.array(inputs), np.array(outputs)

def permute_params(param_dict):
    keys = [ key for key in param_dict.keys() ]
    values = list( itertools.product( *[ param_dict[key] for key in keys ] ) )

    return keys, values

############################################################################################################################
##### SET UP VARIABLES #####################################################################################################
############################################################################################################################

# Get relevant variables
parameters = read_variables()
# Administrative variables
output_dir = parameters["output_dir"]
job = parameters["job"]
max_jobs = parameters["max_jobs"]
# Model parameters
cen_mus = parameters["cen_mus"]
sat_params = parameters["sat_params"]
inner_runs = parameters["inner_runs"]
constant = parameters["constant"]
catalog = parameters["catalog"]
rbins = parameters["rbins"]

# Initial strength parameters
# These don't matter, as they are only needed for the initial creation of the model
central_alignment_strength = 1
satellite_alignment_a = 0.805
satellite_alignment_gamma = -0.029

# Set rbins - Larger max distance means longer run time (from correlation calculations)
#rbins = np.logspace(-1,1.2,15)
#rbins = np.logspace(-1,1.8,29)
rbin_centers = (rbins[:-1]+rbins[1:])/2.0

# Satellite bins
sat_bins = np.logspace(10.5, 15.2, 15)
if catalog == "multidark":
    sat_bins = np.logspace(12.4, 15.5, 15)

# Set up halocat
halocat = CachedHaloCatalog(simname=catalog, halo_finder='rockstar', redshift=0, version_name='halotools_v0p4')
mask_bad_halocat(halocat)

############################################################################################################################
##### RUN ##################################################################################################################
############################################################################################################################

param_dict = {
    "central_alignment_strength":[-1.,-0.5,0,0.5,1.],
    "satellite_alignment_strength":[-1.,-0.5,0,0.5,1.],
    "logMmin":[12.0, 12.5, 13.0],
    "sigma_logM":[0.22, 0.26, 0.3],
    "logM0":[12.3, 12.6, 12.9],
    "logM1":[13.2, 13.5, 13.8],
    "alpha":[0.5, 1.0, 1.5]
}

generate = True
start = time.time()

if generate:

    # Build model instance
    model = build_model_instance(central_alignment_strength, (satellite_alignment_a, satellite_alignment_gamma), sat_bins, 
                                halocat, constant=False, seed=None)

    # Generate data
    # inputs, outputs = generate_training_data(model, rbins, cen_mus, sat_params, halocat, inner_runs=inner_runs, 
    #                                          constant=constant)
    inputs, outputs = generate_training_data_complex(model, rbins, param_dict, halocat, inner_runs=inner_runs, 
                                                     job=job, max_jobs=max_jobs)

    # Save data, making sure to account for if this is a parallelized script
    suffix = ("constant" if constant else "distance_dependent") + (str(job) if not job is None else "")
    #suffix += f"_bins_{rbins[0]}_{rbins[-1]}_{len(rbins)}_inner_loops_{inner_runs}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save( os.path.join( output_dir, f"inputs_{suffix}.npy" ), inputs)
    np.save( os.path.join( output_dir, f"outputs_{suffix}.npy" ), outputs)

    if parameters["job"] is None or parameters["job"] == 1:
        summary_file = open( os.path.join( output_dir, f"summary_{suffix}.txt" ), "w" )
        for key in parameters.keys():
            summary_file.write( f"{key}: {parameters[key]}\n" )
        summary_file.write( f"Time: {time.time()-start}\n" )
        summary_file.close()

print("Time: ", time.time()-start,"\n")

############################################################################################################################
##### TEST #################################################################################################################
############################################################################################################################

# print("Parallel Correlations")
# start= time.time()
# parallel_results = generate_correlations_parallel(model, rbins, halocat)
# print("Time: ", time.time()-start,"\n")

# print("Series Correlations")
# start= time.time()
# series_results = generate_correlations_series(model, rbins, halocat)
# print("Time: ", time.time()-start,"\n")

# Comparisons look good, uncomment to double check
# The multiprocessing parallelization is keeping things in order just fine
# print("Compare Values")
# for i in range(len(parallel_results)):
#     print("Parallel:\t", parallel_results[i])
#     print("Series:\t\t", series_results[i])
#     print("********************")