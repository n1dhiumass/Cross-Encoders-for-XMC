#!/usr/bin/env bash
set -xu

dir=$1
time=${2:-60}
while true
do
rsync -avzi --exclude=.DS_Store --exclude=__pycache__ --max-size='50M' ../../$dir/ blake2:/iesl/canvas/nishantyadav/EfficientPairwiseModels/$dir/
sleep $time
done
