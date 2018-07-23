#! /bin/bash

export PYTHONUNBUFFERED="True"

# Running for REWC

LOG="logs/birds_rewc_vgg_clean`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"
# Test Net
python Birds_vgg16/REWC_vgg16_train.py 
python Birds_vgg16/REWC_vgg16_valid.py 

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"

