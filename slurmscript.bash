#!/usr/bin/bash
# assume 5 hours for each run of about 50k shots

#SBATCH --partition=roma
#SBATCH --job-name=fex2h5
#SBATCH --output=../%j.stdout
#SBATCH --error=../%j.stderr
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3g
#SBATCH --time=0-08:00:00
#SBATCH --gpus 0

PARALLEL=/sdf/home/r/rogersdd/venvs/psana2/bin/parallel

expname=tmox1016823
nshots=-1

# have to repeat this step, since srun isn't cooperating
#cd /sdf/home/r/rogersdd/src/tmo-prefex
#source env.sh

BASE=/sdf/home/r/rogersdd/src/tmo-prefex
#srun -n1 -c64 parallel ./process.sh $nshots $expname ::: `seq 31 94`
#ssh $SLURM_NODELIST $PARALLEL $BASE/process.sh $nshots $expname ::: `seq 31 94`
ssh $SLURM_NODELIST $BASE/process.sh $nshots $expname $@
