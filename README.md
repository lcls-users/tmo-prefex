# Latest update...
Todo, update the .h5 files on /lscratch and then move to /sdf/data/lcls/...  

Thank you Chris O'Grady!  
```python
expname = 'tmox1016823'
runstr = 'r%04i'%(runnum)
segs = ['%03i'%v for v in arange(careful... count files, find with os.path/file etc.)]
dslist = [psana.DataSource(files='/sdf/data/lcls/ds/tmo/%s/xtc/%s-%s-%s-c000.xtc2'%(expname,expname,runstr,ss)) for ss in segs]
```

Chris says also that one can also see what hsd segments are in a file with this command:  

```bash
detnames -i /sdf/data/lcls/ds/tmo/tmoc00123/xtc/tmoc00123-r0022-s010-c000.xtc2
```

# Quick Scripts   
These should go into ./utils folder (not to be confused with ./src/utils.py).  

Developed quick ./src/yield.py in order to quickly show ToF yields in 'mrco\_hsd' in order to tune the position of the chamber to beamline in x(+ toward control room) and y(+ up).  
Hutch coordinates are +z along propagation of x-ray pulses, +x is toward the roll up door, +y is up.  
Plan for yields.py... update to running live monitor with polar plot, and accommodate longer FEX window (xpre,xpost) than the +/-8 used for tmox1016823 (shift 2 x-y tuning).  

# Plan for SUMMIT+  
Run ```fex2h5_minimal.py``` locally in the DRP and send the resulting .h5 files to OLCF.  
Update the quantization vector (serially).  
Reply with the updated quantization vector.  
Refresh plotted histograms based on new quant vecs.  

Add to the plan, for current mode, use old data to start projecting onto eigen-functions from FEX-like thresholded signals.  
Focus on the high intensity runs at the end of the beamtime.  
Also start to do a fully connected feed forward estimator for the peak centroids trained on the eigen coeffecients, versus training on the raw FEX snippets.  

Also, see if the compensation of the ADCs is really an issue or if that is truly only extraneous if we train the eigen functions appropriately.  



#Running on S3DF  
## parallel execution
If using the batchqueue with the ```slurmscript.bash```  
```bash
for r in $(seq 190 1 210); do sbatch slurmscript.bash $r; done
```  
This is set for preemtible and rome nodes in b50.  

## serial execution
OK, moved the below into ```runscript.bash``` and then calling it within slurmscript.bash.   
In that nshots is hard set to 100000.  
The new command to run a sequence of shots is e.g.  
```bash
./runscript.bash $(seq 308 316)
```  
which works, but only sequentially running, not in parallel.  



In case you are running all over from scratch the hits2h5, this is likely a good script-ish thing to do:   
Be sure the create the 'h5files' in the scratch subdirectory.  
```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
export scratchpath=/sdf/data/lcls/ds/tmo/tmox42619/scratch/ryan_output_debug/h5files
export datapath=/sdf/data/lcls/ds/tmo/tmox42619/xtc
export expname=tmox42619
export nshots=100
export configfile=${scratchpath}/${expname}.hsdconfig.h5
python3 ./src/set_configs.py ${configfile}
python3 ./src/hits2h5.py <list of run numbers>
```

