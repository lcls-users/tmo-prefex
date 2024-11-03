# Running on S3DF  

## environment

On S3DF, 

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
# python3 -m venv --system-site-packages ./venv
# . ./venv/bin/activate
# which pip # ensure pip path points to venv!!!
# pip install -r requirements.txt
source ./venv/bin/activate
```

## Basic execution

```bash
ssh psana
# setup env (as above)
fex2h5 <nruns> <runname> <list of run numbers>
```

For additional clues, see `process.sh`, which does some
additional hand-waving before calling fex2h5.

## parallel execution

If using the batchqueue with the ```slurmscript.bash```  
```bash
for r in $(seq 190 1 210); do sbatch slurmscript.bash tmox1016823 $r; done
```  
This is set for milano,
but could maybe use preemtible and rome nodes in b50?
The parallelization here is such that 1 node processes the whole run.

As a side-effect, each node outputs its own h5 file
(all the events it processed).  They can be combined with
```bash
concat_prefex --out run_160.h5 /path/to/...run_160-*.h5
```

# Analyzing data and plotting TOF histogram:

Just 4 `mrco_hsd` detectors, using a 100-bin histogram for the TOF-s.
```bash
tof_corr run_215.h5 0 180 90 270 --nbins 100
```
Note: saves output to `correl.npy`

Very large file (16,000^2 matrix)
```bash
tof_corr run_215.h5 --start 4500 --stop 9000 --nbins 1000
```
Note: will overwrite `correl.npy`

Plotting requires extra packages:
```bash
pip install -r requirements_plot.txt
python plot.py
```

# Sending live data to OLCF/Defiant

Run fex2h5 with --send option:
```bash
ssh sdfdtn003 # need to be on a dtn
. /sdf/home/r/rogersdd/src/tmo-prefex/env.sh
/sdf/home/r/rogersdd/src/tmo-prefex/tmo_cached_push 3001 3000 tmox1016823 <run>
```

This will send batches of results while the h5 files
are being written.  Careful! analysis will halt if the
cache fills up and the receiver doesn't read these messages.

Running analysis on defiant:

1. ask David for how to get a signed cert. to use the API
```bash
message https://defiant/jobs/dtn --yaml mk_hist.yaml
```
see `mk_hist.yaml` for more info.


# Experiment Timeline

## TODO 11/3

- add capture of piranha detector (whole field) to h5
- 2D TOF + piranha cross-correlation
  - ~2GB for 1000 bins per detector
- accumulation mode for 2D hist.

## Latest update...
Todo, update the .h5 files on /lscratch and then move to /sdf/data/lcls/...  

Thank you Chris O'Grady!  
```python
from pathlib import Path

expname = 'tmox1016823'
runstr = 'r%04i'%(runnum)
basedir = Path(f'/sdf/data/lcls/ds/tmo/{expname}/xtc')
dslist = []
for fpath in basedir.iterdir():
    # fname doesn't include parent dir.
    m = re.match(r'(\S+)-r(\d+)-s(\d+)-c000.xtc2', fpath.name)
    if m: # m[1], m[2], m[3] = expname, runstr, ss
        dslist.append(fpath)

#segs = ['%03i'%v for v in arange(careful... count files, find with os.path/file etc.)]
#dslist = [psana.DataSource(files='/sdf/data/lcls/ds/tmo/%s/xtc/%s-%s-%s-c000.xtc2'%(expname,expname,runstr,ss)) for ss in segs]
```

Chris says also that one can also see what hsd segments are in a file with this command:  

```bash
detnames -i /sdf/data/lcls/ds/tmo/tmoc00123/xtc/tmoc00123-r0022-s010-c000.xtc2
```

## Quick Scripts   
These should go into ./utils folder (not to be confused with ./src/utils.py).  

Developed quick ./src/yield.py in order to quickly show ToF yields in 'mrco\_hsd' in order to tune the position of the chamber to beamline in x(+ toward control room) and y(+ up).  
Hutch coordinates are +z along propagation of x-ray pulses, +x is toward the roll up door, +y is up.  
  - Is this a right-handed coordinate system?

Plan for yields.py... update to running live monitor with polar plot, and accommodate longer FEX window (xpre,xpost) than the +/-8 used for tmox1016823 (shift 2 x-y tuning).  

## Plan for SUMMIT+  

Run ```fex2h5``` locally in the DRP and send the resulting
.h5 files to OLCF (--send argument).

Accumulate each batch of data sent into the TOF histograms.
Reply with the updated histograms.

Add to the plan, for current mode, use old data to start projecting onto eigen-functions from FEX-like thresholded signals.  
Focus on the high intensity runs at the end of the beamtime.  
Also start to do a fully connected feed forward estimator for the peak centroids trained on the eigen coeffecients, versus training on the raw FEX snippets.  

Also, see if the compensation of the ADCs is really an issue or if that is truly only extraneous if we train the eigen functions appropriately.  


