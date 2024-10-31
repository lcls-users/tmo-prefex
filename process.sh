#!/bin/bash

cd /sdf/home/r/rogersdd/src/tmo-prefex
source env.sh

nshots=$1
expname=$2
runnum=$3

datapath=/sdf/data/lcls/ds/tmo/$expname/xtc
printf -v runstr "r%04d" $runnum

# seems to include s000 .. s019
if [ -f $datapath/$expname-$runstr-s000-c000.xtc2 ]; then
    python3 ./src/fex2h5.py $nshots $expname $runnum
else
    echo "XTC2 file not found for run $expname:$runstr"
fi
