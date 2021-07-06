import xarray as xr
import pathlib
from pathlib import Path
import numpy as np
import cv2
import h5netcdf
import h5py
import glob
import re
import os
import sys
from data_prep_utils import pad_data

"""
Script for making downsampled data from the original.

About Data:

* Native files: native files are found in .nc files (e.g. spherical001.nc), when converted 
  to numpy arrays have a shape of (     1,        8,           201,          180,    360 ) 
                                  (# samples, # variables, # radial layers, height, width)

* Ground truth data: the ground truth data (used by the descriminator) is scaled to 60%
  of the original native file height and width (the latitude and longitude). The final shape
  of a ground truth data sample (e.g. x_truth_001.npy) is (1, 4, 198, 108, 216)
        * Only the temperature, velocity components vx, vy, and vz are include as variables.
        The thermal conductivity and expansivity variables had degraded data from radial layer 
        190 onwards. The spin transition induced density anomaly variable also had inconsitencies.
        The temperature anomaly could be worked in at a later date as it showed no problems.
        * For the variables included, the first radial index (radius 3485) and last index (radius 6371) 
        were corrupt and dropped from all timesteps. The output from the generator has a radial dim of
        198, and thus another radial layer was dropped
        * Empirically, it was found that the 60% downscaling did not remove much information, but 
        increased the speed of training.

* Data for generator: the input data for the generator (e.g. x_001.npy) has a shape of (1, 4, 198, 14, 27).
  It is 8x smaller than the ground truth data.
        * Note: the x_001.npy arrays are stored with shape (1, 4, 198, 14, 27), but the input
        to the generator is of shape (1, 4, 30, 20, 10)
"""

#!#!#!#!#!#!#!#!# HELPER FUNCTIONS #!#!#!#!#!#!#!#!#

def load_data(data_path : pathlib.PosixPath):
    """Load a .nc file and ouput a numpy array with the appropriate
    ordering of dimensions (# samples, # variables, # radial layers, height, width)
    """
    
    data = xr.open_dataset(data_path,)

    # create a list of all the variables.
    var_list = list(data.keys())
    
    radial_array = data.r.values

    # stack all the variables together, about the last dimension
    x = np.stack([data[var_name].values for var_name in var_list], axis=-1)

    # move height (latitudes) before width (longitudes)
    # final output will have shape [no_samples, no_variables, radial_index, latitude, longitude]
    return np.expand_dims(np.moveaxis(x, source=[3, 0], destination=[0,2]),0), var_list, radial_array


def downscale(x, downscale_percent, cv2_interp=cv2.INTER_CUBIC):
    """Take earth mantle numpy array and downscale by a percentage. 
    """

    # shape of x should be [s, v, r, h, w]
    # only downsample if downscale_percent < 1
    h = x.shape[3]
    w = x.shape[4]
    
    if downscale_percent != 1:
        # downscale data
        width = int(w * downscale_percent)
        height = int(h * downscale_percent)
        dim = (width, height)
        data_list = []
        
        # iterate through each variable
        for var_index in range(x.shape[1]):
            temp_list = []
            # iterate trhough each radius
            for radius_index in range(x[0,var_index,:,:,:].shape[0]):
                temp_list.append(cv2.resize(x[0,var_index,radius_index,:,:], dim, interpolation=cv2_interp))

            temp_array = np.stack(temp_list, axis=0)
            data_list.append(temp_array)

        # stack data_list into a np array
        return np.expand_dims(np.stack(data_list, axis=0,),0)
    else:
        return x


#!#!#!#!#!#!#!#!# RUN SCRIPT #!#!#!#!#!#!#!#!#

### Set parameters ###
# Note: changing these parameter constants
# may prevent the models from training
# as these constanst were selected to optimize memory


# make your root_dir the current working directory
root_dir = Path.cwd() # set the root directory as a Pathlib path

folder_processed_data = root_dir / 'data/processed/'
folder_input_data = root_dir / 'data/processed/input' # 8x downsampled data
folder_truth_data = root_dir / 'data/processed/truth' # "original" data (60% of native file h & w)

# make processed data folder if they don't exist already
Path(folder_processed_data).mkdir(parents=True, exist_ok=True)
Path(folder_input_data).mkdir(parents=True, exist_ok=True)
Path(folder_truth_data).mkdir(parents=True, exist_ok=True)

folder_raw_data = root_dir / 'data/raw/' # raw data folder that holds the .nc files
