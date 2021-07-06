import xarray as xr
import pathlib
from pathlib import Path
import numpy as np

import matplotlib
# run matplotlib without display
# https://stackoverflow.com/a/4706614/9214620
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import glob
import re
import os
import sys

#!#!#!#!#!#!#!#!# PLOTTING FUNCTION AND OTHER HELPER FUNCTIONS #!#!#!#!#!#!#!#!#

def plot_radial_layer(data, r_index, time_step='001', save_img=False, save_folder_name='radial_layer_img',):
    
    # if save folder doesn't exist, create it
    if save_img:
        img_folder = Path(save_folder_name)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14,8), dpi=40)
    
    color_scheme='inferno'

    for variable, ax in zip(list(data.keys()), axes.flat):
        ax.pcolormesh(data[variable].values[:,r_index,:], cmap=color_scheme)
        ax.set_title(f"{variable}")
        min_val = np.min(data[variable].values[:,r_index,:])
        max_val = np.max(data[variable].values[:,r_index,:])

        print_text = f"timestep = {time_step}\nr_index = {r_index}\nmin = {min_val:.2E}\nmax = {max_val:.2E}"

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.text(
            (x_max - x_min) * 0.03 + x_min,
            y_max - (y_max - y_min) * 0.05,
            print_text,
            fontsize=18,
            fontweight="normal",
            verticalalignment="top",
            color='white',
            horizontalalignment="left",
            bbox={"facecolor": "gray", "alpha": 0.0, "pad": 6},
        )
    
    fig.tight_layout(w_pad=2.0, h_pad=2.0)

    if save_img:
        save_name = img_folder / f'{time_step}_{r_index}.jpg'
        plt.savefig(save_name,)
        plt.close('all')
    else:
        plt.show()


#!#!#!#!#!#!#!#!# CREATE THE IMAGES #!#!#!#!#!#!#!#!#


# make your root_dir the current working directory
if Path.cwd() == Path('/home/tim/Documents/deep-compression'):
    root_dir = Path.cwd() # use if you are on local computer

    # create save folder if not already in place
    save_folder_name = 'radial_layer_img'
    pathlib.Path(root_dir / save_folder_name).mkdir(parents=True, exist_ok=True)

else: # assume we're on HPC
    root_dir = Path('/home/tvhahn/projects/def-mechefsk/tvhahn/deep-compression') # set the root directory as a Pathlib path
    scratch_dir = Path('/home/tvhahn/scratch/earth') # scratch directory where we'll save images

    # create save folder if not already in place
    save_folder_name = scratch_dir / 'radial_layer_img'
    pathlib.Path(save_folder_name).mkdir(parents=True, exist_ok=True)



folder_raw_data = root_dir / 'data/raw/' # raw data folder that holds the .nc files

# get the file name as input
file_name = sys.argv[1]
print("file_name input:", file_name)

# get time step
time_step = re.findall('[0-9]+', str(file_name))[-1]

# load data file
data = xr.open_dataset(folder_raw_data / file_name)

# get no. radial layers
for r_index in range(data.r.values.shape[0]):
    plot_radial_layer(data, r_index, time_step, save_img=True, save_folder_name=save_folder_name,)

