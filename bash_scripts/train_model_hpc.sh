#!/bin/bash
#SBATCH --account=rrg-mechefsk
#SBATCH --gres=gpu:t4:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=14000M      # memory per node
#SBATCH --time=0-00:15      # time (DD-HH:MM)
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
# cd
# tensorboard --logdir=scratch/earth-mantle-surrogate/models/interim/logs --samples_per_plugin images=250 --host 0.0.0.0 &

# begin training
python $PROJECT_DIR/src/models/train_model.py \
    --path_data $SLURM_TMPDIR/data/processed \
    --proj_dir $PROJECT_DIR \
    --checkpoint  2021_11_03_102524 \
    --model_time_suffix foo \
    --batch_size 1 \
    --var_to_include 1 \
    --learning_rate 1e-4 \
    --critic_iterations 5 \
    --num_epochs 500 \
    --lambda_gp 10 \
    --gen_pretrain_epochs 5 \
    # --cat_noise
