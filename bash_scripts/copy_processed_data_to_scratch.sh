#!/bin/bash
DIR="~/scratch/earth-mantle-surrogate"
if [ ! -d "$DIR" ]; then
    echo "earth-mantle-surrogate folder in scratch not exist"
    mkdir ~/scratch/earth-mantle-surrogate
fi

cp -r ./data/processed ~/scratch/earth-mantle-surrogate
