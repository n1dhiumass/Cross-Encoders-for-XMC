#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

fileid="1BBTue5Vmr3MteGcse-ePqplWjccqm9_A"
filename="zeshelmodels.tar.bz2"
curl -c ./cookie -s -L "https://drive.google.com/u/0/uc?id=1BBTue5Vmr3MteGcse-ePqplWjccqm9_A&export=download" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
