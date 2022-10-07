#!/usr/bin/env bash
set -xu

dir=$1
size=${2:-50000}
minsize=${3:-0}
#time=${2:-60}
#while true
#do
rsync -avzi --max-size=${size}M --min-size=${minsize}M --exclude wandb gypsum:/mnt/nfs/work1/mccallum/nishantyadav/EfficientPairwiseModels/$dir/ /iesl/canvas/nishantyadav/EfficientPairwiseModels/$dir/
#sleep $time
#done
