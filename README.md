# What is broke...  

When processing mroe than 1000 shots (like 2000) this breaks likely because we hit the raw wave sampling...   

```bash
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
writing to /sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/sdfiana008/hits.tmox1016823.run_076.h5
Traceback (most recent call last):
  File "/sdf/home/c/coffee/tmo-prefex/./src/fex2h5.py", line 249, in <module>
    main(nshots,expname,runnums,scratchdir)
  File "/sdf/home/c/coffee/tmo-prefex/./src/fex2h5.py", line 235, in main
    s >> write_out(ports, outname)
  File "/sdf/home/c/coffee/tmo-prefex/venv/lib/python3.9/site-packages/stream/core.py", line 89, in __rshift__
    return Stream.pipe(self, outpipe)
  File "/sdf/home/c/coffee/tmo-prefex/venv/lib/python3.9/site-packages/stream/core.py", line 76, in pipe
    return out(iter(inp))    # connect streams
  File "/sdf/home/c/coffee/tmo-prefex/venv/lib/python3.9/site-packages/stream/core.py", line 189, in __call__
    return self.consumer(iterator, *self.args, **self.kws)
  File "/sdf/home/c/coffee/tmo-prefex/./src/fex2h5.py", line 175, in write_out
    for batch in inp:
  File "/sdf/home/c/coffee/tmo-prefex/src/combine_port.py", line 116, in <lambda>
    save_dd_batch = lambda u: map_dd(u, save_batch)
  File "/sdf/home/c/coffee/tmo-prefex/src/combine_port.py", line 76, in map_dd
    ans[k][k2] = fn(v2)
  File "/sdf/home/c/coffee/tmo-prefex/src/combine_port.py", line 111, in save_batch
    raw = np.hstack([waves[i].raw for i in raw_idx]),
  File "<__array_function__ internals>", line 200, in hstack
  File "/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps-4.6.3/lib/python3.9/site-packages/numpy/core/shape_base.py", line 370, in hstack
    return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)
  File "<__array_function__ internals>", line 200, in concatenate
ValueError: need at least one array to concatenate
(venv) [coffee@sdfiana008 tmo-prefex]$ 
```

# Latest update...

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

