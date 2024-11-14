#!/usr/bin/bash
#SBATCH --job-name=fex2h5
#SBATCH --output=%x-%j.stdout
#SBATCH --error=%x-%j.stderr
#SBATCH --reservation lcls:earlyscience
#SBATCH --partition=milano
#SBATCH --nodes 1
#SBATCH --ntasks 120
#SBATCH --cpus-per-task=1
#SBATCH -A lcls:tmox1016823
#SBATCH -t 120
#SBATCH --exclusive


if [ $# -lt 2 ]; then
  echo "Usage: sbatch $0 <expname> <runnum> [--dial tcp://addr:port]"
  exit 1
fi

expname=$1
runnum=$2
detectors=gmd,spect,hsd

PARALLEL=/sdf/home/r/rogersdd/venvs/psana2/bin/parallel
BASE=/sdf/home/r/rogersdd/src/tmo-prefex

cd /sdf/home/r/rogersdd/src/tmo-prefex
source env.sh

st=`date +%s`

printf -v runstr "r%04d" $runnum

#datapath=/sdf/data/lcls/ds/tmo/$expname/xtc
# seems to include s000 .. s019
#if [ -f $datapath/$expname-$runstr-s000-c000.xtc2 ]; then
# If do right, no can defense.
time mpirun xtc2h5 $@
#else
#    echo "XTC2 file not found for run $expname:$runstr"
#fi

en=`date +%s`
echo "Completed processing $nshots from $expname:$runnum in $((en-st)) sec."
