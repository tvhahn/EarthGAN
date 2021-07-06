#!/bin/bash
PROJECT_DIR=$1

cd
module load python/3.8
source ~/earth/bin/activate

cd $PROJECT_DIR
python $PROJECT_DIR/src/data/make_dataset.py $PROJECT_DIR/data/raw $PROJECT_DIR/data/interim $PROJECT_DIR/data/processed