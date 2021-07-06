#!/bin/bash
#SBATCH --time=00:10:00 # 10 minutes
#SBATCH --array=1-251
#SBATCH --mem=2G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $
## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays

echo "Starting task $SLURM_ARRAY_TASK_ID"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" file_list)

# load module and create temp virtual env
module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index xarray==0.16.2 scipy==1.6.0 netCDF4==1.5.6
pip install --no-index h5netcdf==0.7.4 matplotlib==3.3.4

python make_all_timestep_images.py $DIR