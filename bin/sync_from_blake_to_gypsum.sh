#!/usr/bin/env bash
set -xu

dir=$1
#time=${2:-60}
#while true
#do
rsync -avzi --exclude "slurm*.out" --exclude=__pycache__ --exclude=jupyter-notebooks /iesl/canvas/nishantyadav/EfficientPairwiseModels/$dir/ gypsum:/mnt/nfs/work1/mccallum/nishantyadav/EfficientPairwiseModels/$dir/
#sleep $time
#done
