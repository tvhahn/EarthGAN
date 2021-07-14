#!/bin/bash
cwd=$(pwd)
cd
tensorboard --logdir=scratch/earth-mantle-surrogate/models/interim/logs --host 0.0.0.0 &