#!/usr/bin/bash
# assume 5 hours for each run of about 50k shots

#SBATCH --partition=roma
#
#SBATCH --job-name=hits2h5_min
#SBATCH --output=../output-%j.stdout
#SBATCH --error=../output-%j.errout
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8g
#
#SBATCH --time=0-08:00:00
#
#SBATCH --gpus 0

host=`hostname`
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
. /sdf/home/r/rogersdd/src/tmox1016823/venv/bin/activate
export scratchpath=/sdf/data/lcls/ds/tmo/tmox1016823/scratch/$USER/h5files/$host
test -d $scratchpath || mkdir -p $scratchpath
export expname=tmox1016823
export nshots=100000
export configfile=$scratchpath/$expname.hsdconfig.h5
export datapath=/sdf/data/lcls/ds/tmo/$expname/xtc
printf -v runstr "r%04d" $1

# seems to include s000 .. s019
if [ -f $datapath/$expname-$runstr-s000-c000.xtc2 ]; then
	python3 ./src/set_configs.py $configfile
	python3 ./src/hits2h5_minimal.py $1
else
	echo "XTC2 file not found for run $expname:$runstr"
fi

