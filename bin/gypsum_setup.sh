#!/usr/bin/env bash

set -x

export ROOT_DIR=`pwd`
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/nfs/work1/mccallum/nishantyadav/anaconda3/lib

set +x