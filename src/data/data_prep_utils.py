import xarray as xr
import pathlib
from pathlib import Path
import numpy as np
import cv2


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


def pad_data_numpy(x, pad_top_bot=0, pad_sides=0):
    """pad the x data along on the sides
    
    Parameters
    ===========
    x : numpy array
        Earth mantle array
    
    pad_top_bot : int
        Pad amount on the top and bottom (along latitudinal axis)
        
    pad_sides : int
        Pad amount on the sides (along longitudinal axis)

    Returns
    ===========
    x : numpy array
        Numpy array with height increased by 2*pad_top_bot
        and width increased by 2*pad_sides
    """

    if pad_sides > 0:
        x = np.concatenate((x[:,:,:,:,-pad_sides:], x, x[:,:,:,:,:pad_sides]), axis=-1)
    
    if pad_top_bot > 0:
        x = np.concatenate((np.flip(x, [3])[:,:,:,-pad_top_bot:,:], # mirror array and select top rows
                         x, 
                         np.flip(x, [3])[:,:,:,:pad_top_bot,:]), # mirror array and select bottom rows
                      axis=-2) # append along longitudinal (left-right) axis
        
    return x


def keep_variables(x, original_var_list, var_list_to_keep):
    """Keep select variable from the earth mantle data"""
    var_index_list = []
    for i, v in enumerate(original_var_list):
        if v in var_list_to_keep:
            var_index_list.append(i)
    return x[:,np.array(var_index_list),:,:,:]


def create_truth_and_downsampled_x(file, downscale_percent_1=0.60, downscale_percent_2=1/8.0):
    """Create the "truth" (e.g. x_truth_001) and "downsampled (e.g. x_001) array 
    from an .nc file."""

    # original shape of x is (1, 8, 201, 180, 360)
    x_truth, var_list, radial_array = load_data(file)

    # select only the first 4 variables.
    # ['temperature','vx','vy','vz']
    var_list_to_keep = var_list[0:4]
    x_truth = keep_variables(x_truth, var_list, var_list_to_keep)

    # drop r_index 0,1, 200
    # index 0 and 200 are corrupt
    # index 1 is dropped to get final radial dim to 198
    r_index_array = np.arange(0, len(radial_array))[2:-1]
    x_truth = x_truth[:,:,r_index_array,:,:]

    # downscale the x_truth by 40% to get final x_truth (needed for discriminator)
    # shape of x_truth will be (1, 4, 198, 108, 216)
    x_truth = downscale(x_truth, downscale_percent_1, cv2_interp=cv2.INTER_CUBIC)

    # pad the x_truth such that it can be downscaled without rounding
    x = pad_data_numpy(x_truth, pad_top_bot=2, pad_sides=0)

    # downscale x to final shape of (1, 4, 198, 14, 27)
    x = downscale(x, downscale_percent_2, cv2_interp=cv2.INTER_CUBIC)

    return x_truth, x


def min_max_array(x):
    # get the min/max value for each variable
    min_vals = np.array([np.min(x[0,i,:,:,:]) for i in range(np.shape(x)[1])])
    max_vals = np.array([np.max(x[0,i,:,:,:]) for i in range(np.shape(x)[1])])
    return min_vals, max_vals


def scaler(x, min_val_array, max_val_array):
    """scaler data between 0 and 1"""
    # get the shape of the array
    # s = no. samples (should be just 1)
    # h = height (latitude)
    # r = radius
    # w = width (longitude)
    # v = no. variables (channels)
    # shape of x should be [s, v, r, h, w]
    s, v, r, h, w = np.shape(x)
    
    for radius in range(r):
        for var in range(v):
            x[0,var,radius,:,:] = np.divide((x[0,var,radius,:,:] - min_val_array[var]), np.abs(max_val_array[var] - min_val_array[var]))
           
    return x
