#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --job-name=fex_peak
#SBATCH --output=/sdf/home/b/bmencer/github/tmo-prefex/src/output/output-%j.txt
#SBATCH --error=/sdf/home/b/bmencer/github/tmo-prefex/src/output/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=80g
#SBATCH --time=0-40:0:00
##SBATCH --gpus 1

export datapath="/sdf/data/lcls/ds/tmo/tmox1016823/xtc"
export expname="tmox1016823"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
python /sdf/home/b/bmencer/github/tmo-prefex/src/fex_peakfinding.py