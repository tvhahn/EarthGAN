#!/bin/bash
#SBATCH --time=01:00:00 # 1 hr
#SBATCH --mem=4G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

cd
module load python/3.8
source ~/earth/bin/activate

cd $PROJECT_DIR
python $PROJECT_DIR/src/data/make_dataset.py data/raw data/interim data/processed