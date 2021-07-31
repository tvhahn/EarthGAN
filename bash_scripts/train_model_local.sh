#!/bin/bash
PROJECT_DIR=$1

cd $PROJECT_DIR

# must pass the location of process data to train_model.py
# can pass optional checkpoint folder name. e.g.:
#   python train_model.py ./data/processed -c 2021_07_15_093623
# train_model.py will then look for the most recent checkpoint .pt file
# in the checkpoint folder and load the checkpoint before model training
python $PROJECT_DIR/src/models/train_model.py ./data/processed