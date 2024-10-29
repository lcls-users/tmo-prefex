#!/usr/bin/bash

#SBATCH --partition=roma
#SBATCH --account=lcls:tmox1016823
#SBATCH --job-name=fex2h5
#SBATCH --output=../output-%j.stdout
#SBATCH --error=../output-%j.errout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=0-05:00:00
#SBATCH --mail-user=coffee@slac.stanford.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --gpus 0

export logpath=$HOME/slurmout
if ! [ -f $logpath ]; then
	mkdir -p $HOME/slurmout
fi

export expname=$1
export runnum=$2
export runstr="r$(printf "%04i" $2)"
export nshots=5000
#export nshots=250000
echo "trying to run $expname $runstr $nshots"
export datapath=/sdf/data/lcls/ds/tmo/$expname/xtc
export finalpath=/sdf/data/lcls/ds/tmo/$expname/scratch/$USER/h5files/$runstr
export scratchpath=/lscratch/$USER/h5files/$runstr
export configpath=/lscratch/$USER/configs/$runstr
if ! [ -f $scratchpath ]; then
	echo "creating $scratchpath"
	mkdir -p $scratchpath
fi
if ! [ -f $configpath ]; then
	echo "creating $configpath"
	mkdir -p $configpath
fi
if ! [ -f $finalpath ]; then
	echo "creating $finalpath"
	mkdir -p $finalpath
fi

if [ -f ${datapath}/${expname}-${runstr}-s000-c000.xtc2 ]; then
	source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
	python3 $HOME/tmo-prefex/src/fex2h5.py $nshots $expname $runnum
	echo "syncing $scratchpath to $finalpath \nand then removing $scratchpath"
	rsync -pruv $scratchpath/* $finalpath/
	rsync -pruv $configpath/* $finalpath/
	rm -rf $scratchpath
else
	echo "XTC2 file not found for run ${expname}:${runstr}"
fi

