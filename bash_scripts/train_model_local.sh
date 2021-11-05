#!/bin/bash
PROJECT_DIR=$1

cd $PROJECT_DIR

# must pass the location of process data to train_model.py
# can pass optional checkpoint folder name. e.g.:
#   python train_model.py ./data/processed -c 2021_07_15_093623
# train_model.py will then look for the most recent checkpoint .pt file
# in the checkpoint folder and load the checkpoint before model training
python $PROJECT_DIR/src/models/train_model.py \
    --path_data ./data/processed \
    --proj_dir $PROJECT_DIR \
    --batch_size 1 \
    --var_to_include 4 \
    --learning_rate 1e-4 \
    --critic_iterations 5 \
    --num_epochs 1000 \
    --lambda_gp 10 \
    --gen_pretrain_epochs 20 \
    --model_time_suffix var4 \
    --checkpoint  2021_11_04_191811_var4 \
    # --cat_noise