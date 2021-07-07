#!/bin/bash
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

# python $PROJECT_DIR/src/data/make_dataset.py data/raw data/interim data/processedta/processed