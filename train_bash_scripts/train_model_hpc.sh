#!/bin/bash
#SBATCH --account=rrg-mechefsk
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=7   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M      # memory per node
#SBATCH --time=0-01:20      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1

source ~/earth/bin/activate

# copy processed data from scratch to the temporary directory used for batch job
# this will be much faster as the train_model.py rapidly access the training data
mkdir $SLURM_TMPDIR/data
cp -r ~/scratch/earth-mantle-surrogate/processed $SLURM_TMPDIR/data

# load tensorboard
cd
tensorboard --logdir=scratch/earth-mantle-surrogate/models/interim/logs --host 0.0.0.0 &

# begin training
python $PROJECT_DIR/src/models/train_model.py $SLURM_TMPDIR/data/processed -c 2021_07_15_114841