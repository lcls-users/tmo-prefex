RUN=$@
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
export scratchpath=/sdf/data/lcls/ds/tmo/tmox42619/scratch/ryan_output_debug/h5files
export datapath=/sdf/data/lcls/ds/tmo/tmox42619/xtc
export expname=tmox42619
export nshots=100000
export configfile=${scratchpath}/${expname}.hsdconfig.h5
python3 ./src/set_configs.py ${configfile}
for r in $RUN; do
	python3 ./src/hits2h5_minimal.py $r
done

