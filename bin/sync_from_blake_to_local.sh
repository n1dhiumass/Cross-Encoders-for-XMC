#!/usr/bin/env bash
set -xu

dir=$1
time=${2:-60}
size=${3:-1}
while true
do
rsync -avzi --exclude "wandb" --exclude "code" --max-size=${size}M blake2:/iesl/canvas/nishantyadav/EfficientPairwiseModels/code/cross-encoder-xmc/$dir/ $dir/
sleep $time
done
