# Latest update...

assuming 10000 shots for a quick run.  
```bash
for r in $(seq 70 79); do sbatch slurmscript.bash tmox1016823 $r 10000;done
```

If you want to see the output and run from the interactive terminal for e.g. 5000 shots for e.g. run 85.  
```bash
bash slurmscript.bash tmox1016823 85 10000
```

For a future when event building is not needed...  
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
And the answer is that only one hsd channel is in a segment.  

# Quick Scripts   
These should go into ./utils (or in David's /cmd) folder (not to be confused with ./src/utils.py).  

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

# Running on S3DF  
## parallel execution
If using the batchqueue with the ```slurmscript.bash```  
```bash
for r in $(seq 190 1 210); do sbatch slurmscript.bash $r; done
```  
This is set for preemtible and rome nodes in b50.  

# Working with already pre-processed h5 fileson s3df  
```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
conda deactivate
conda activate h5-1.0.1
```


