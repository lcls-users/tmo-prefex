#!/usr/bin/bash
#SBATCH --partition=roma
#SBATCH --job-name=fex2h5
#SBATCH --output=%x-%j.stdout
#SBATCH --error=%x-%j.stderr
#SBATCH --ntasks 32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3g
#SBATCH --time=0-02:00:00
#SBATCH --gpus 0

expname=tmox1016823
nshots=0
runnum=215

PARALLEL=/sdf/home/r/rogersdd/venvs/psana2/bin/parallel
BASE=/sdf/home/r/rogersdd/src/tmo-prefex
datapath=/sdf/data/lcls/ds/tmo/$expname/xtc

# have to repeat this step, since srun isn't cooperating
#cd /sdf/home/r/rogersdd/src/tmo-prefex
#source env.sh

#srun -n1 -c64 parallel ./process.sh $nshots $expname ::: `seq 31 94`

# Note: can use parallel if you ask for lots of CPU-s
#ssh $SLURM_NODELIST $PARALLEL $BASE/process.sh $nshots $expname ::: `seq 31 94`

#ssh $SLURM_NODELIST $BASE/process.sh $nshots $expname $@


cd /sdf/home/r/rogersdd/src/tmo-prefex
source env.sh

st=`date +%s`

printf -v runstr "r%04d" $runnum

# seems to include s000 .. s019
if [ -f $datapath/$expname-$runstr-s000-c000.xtc2 ]; then
    # If do right, no can defense.
    time mpirun fex2h5 $nshots $expname $runnum
else
    echo "XTC2 file not found for run $expname:$runstr"
fi

en=`date +%s`
echo "Completed processing $nshots from $expname:$runnum in $((en-st)) sec."
