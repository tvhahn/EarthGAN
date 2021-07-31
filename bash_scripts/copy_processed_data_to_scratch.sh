#!/bin/bash
DIR="~/scratch/EarthGAN"
if [ ! -d "$DIR" ]; then
    echo "EarthGAN folder in scratch not exist"
    mkdir ~/scratch/EarthGAN
fi

cp -r ./data/processed ~/scratch/EarthGAN
