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

# check that all the appropriate data folders are created
if [ ! -d "data/raw" ]; then
    echo "Ensure data downloaded"
fi

if [ ! -d "data/interim" ]; then
    mkdir data/interim
fi

if [ ! -d "data/processed" ]; then
    mkdir data/processed
fi

python $PROJECT_DIR/src/data/make_dataset.py data/raw data/interim data/processed