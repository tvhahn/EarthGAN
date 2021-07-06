# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from pathlib import Path
import numpy as np
import re
from src.data.data_prep_utils import (
    create_truth_and_downsampled_x,
    min_max_array,
    scaler,
)


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

#!#!#!#!#!#!#!#!# RUN SCRIPT #!#!#!#!#!#!#!#!#


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("interim_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, interim_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # folder_raw_data = root_dir / 'data/raw/' # raw data folder that holds the .nc files
    folder_raw_data = (
        root_dir / input_filepath
    )  # raw data folder that holds the .nc files

    folder_input_data_interim = (
        root_dir / interim_filepath / "input"
    )  # 8x downsampled data - unscaled
    folder_truth_data_interim = (
        root_dir / interim_filepath / "truth"
    )  # "original" data (60% of native file h & w) - unscaled

    folder_input_data = root_dir / output_filepath / "input"  # final data scaled
    folder_truth_data = root_dir / output_filepath / "truth"  # final data scaled

    # make processed data folder if they don't exist already
    Path(folder_input_data_interim).mkdir(parents=True, exist_ok=True)
    Path(folder_truth_data_interim).mkdir(parents=True, exist_ok=True)
    Path(folder_input_data).mkdir(parents=True, exist_ok=True)
    Path(folder_truth_data).mkdir(parents=True, exist_ok=True)

    # get a list of file names
    file_list = list(folder_raw_data.rglob("*.nc"))

    # instantiate min/max lists that will be used to later
    # scale the data
    min_list_input = []
    min_list_truth = []
    max_list_input = []
    max_list_truth = []

    # create all interim data (not yet scaled)
    for i, file in enumerate(file_list):
        # get time step
        time_step = re.findall("[0-9]+", str(file))[-1]
        x_save_name = f"x_{time_step}.npy"
        x_truth_save_name = f"x_truth_{time_step}.npy"

        x_truth, x = create_truth_and_downsampled_x(file)

        np.save(folder_input_data_interim / x_save_name, x)
        np.save(folder_truth_data_interim / x_truth_save_name, x_truth)

        min_vals, max_vals = min_max_array(x)
        min_list_input.append(min_vals)
        max_list_input.append(max_vals)

        min_vals, max_vals = min_max_array(x_truth)
        min_list_truth.append(min_vals)
        max_list_truth.append(max_vals)

    min_array_input = np.min(np.array(min_list_input), axis=0)
    max_array_input = np.max(np.array(max_list_input), axis=0)

    min_array_truth = np.min(np.array(min_list_truth), axis=0)
    max_array_truth = np.max(np.array(max_list_truth), axis=0)

    # scale all data and save in processed folder
    for i, file in enumerate(file_list):
        time_step = re.findall("[0-9]+", str(file))[-1]
        x_save_name = f"x_{time_step}.npy"
        x_truth_save_name = f"x_truth_{time_step}.npy"

        x = scaler(
            np.load(folder_input_data_interim / x_save_name),
            min_array_input,
            max_array_input,
        )
        np.save(folder_input_data / x_save_name, x)

        x_truth = scaler(
            np.load(folder_truth_data_interim / x_truth_save_name),
            min_array_truth,
            max_array_truth,
        )
        np.save(folder_truth_data / x_truth_save_name, x_truth)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # establish folders for holding raw, interim (un-scaled data), and final processed
    root_dir = Path(__file__).resolve().parents[2]

    main()
