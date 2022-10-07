#!/usr/bin/env bash
set -xu

dir=$1
#time=${2:-60}
#while true
#do
rsync -avzi gypsum:/mnt/nfs/scratc1/nishantyadav/$dir/ /iesl/canvas/nishantyadav/EfficientPairwiseModels/$dir/
#sleep $time
#done
