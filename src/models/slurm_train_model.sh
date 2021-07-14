#!/bin/bash
#SBATCH --account=rrg-mechefsk
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M      # memory per node
#SBATCH --time=0-00:20      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECTDIR=~/projects/def-mechefsk/tvhahn/earth-mantle-surrogate

source ~/earth/bin/activate

# copy processed data from scratch to the temporary directory used for batch job
mkdir $SLURM_TMPDIR/data
cp -r ~/scratch/earth-mantle-surrogate/processed $SLURM_TMPDIR/data

# begin training
python $PROJECTDIR/src/models/train_model.py $SLURM_TMPDIR/data/processed