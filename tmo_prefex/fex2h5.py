#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import List,Dict,Optional,Tuple,Union
from collections.abc import Iterator
from pathlib import Path
import sys
import re
import os

import h5py
import psana
import numpy as np
from stream import (
    stream, source, sink,
    take, Source, takei, seq, gseq,
    chop, map, filter, takewhile,
)

from .Config import Config
from .stream_utils import variable_chunks

from .Hsd import HsdConfig, WaveData, FexData, run_hsds, setup_hsds, save_hsd
from .Ebeam import EbeamConfig, EbeamData, setup_ebeams, run_ebeams, save_ebeam
from .Gmd import GmdConfig, GmdData, setup_gmds, run_gmds, save_gmd
from .Spect import SpectConfig, SpectData, setup_spects, run_spects, save_spect

# Some types:
DetectorID   = Tuple[str, int] # ('hsd', 22)
DetectorData = Union[WaveData, FexData, GmdData, EbeamData, SpectData]
EventData    = Dict[DetectorID, DetectorData]

def save_fex(run, params):
    return setup_hsds(run, params, default_fex)

# Plugin system for detector types:
detector_configs = {
    # NOTE: defaults to fex-type hsd setup
    'hsd': (setup_hsds, run_hsds, save_hsd),
    'ebeam': (setup_ebeams, run_ebeams, save_ebeam),
    'gmd': (setup_gmds, run_gmds, save_gmd),
    'spect': (setup_spects, run_spects, save_spect),
}

from stream_utils import split, xmap
from combine import batch_data, Batch

@source
def run_events(run, start_event=0):
    for i, evt in enumerate(run.events()):
        yield i+start_event, evt

@sink
def write_out(inp: Iterator[Batch], outname: str) -> None:
    for i, batch in enumerate(inp):
        name = f"{outname}.{i}.h5"
        print('writing batch %i to %s'%(i,name))
        with h5py.File(name,'w') as f:
            batch.write_h5(f)

def process_all(tps):
    for dtype in tps:
        for k, v in dtype.items():
            w = v.process()
            if w != v:
                print(f"{dtype} - incorrect return from process()")
    return tps

@stream
def live_events(events, max_consecutive=100):
    errs = []
    consecutive = 0
    for ev,dets in enumerate(events):
        failed = [i for i,d in enumerate(dets) if d is None]
        if len(failed) == 0:
            yield dets
            consecutive = 0
            continue

        consecutive += 1
        errs.append(failed)
        if consecutive >= max_consecutive:
            break
    
    if consecutive >= max_consecutive:
        print(f"Stopping early at event {ev} after {consecutive} errors")
        print("Failed detector list:")
        for e in errs:
            print(errs[-consecutive:])
    print(f"Processed {ev} events with {len(errs)} dropped.")

def main(nshots:int, expname:str, runnums:List[int], scratchdir:str):
    #######################
    #### CONFIGURATION ####
    #######################
    cfgname = '%s/%s.%s.configs.yaml'%(scratchdir,expname,os.environ.get('USER'))
    inp_cfg = 'config.yaml'
    if Path(inp_cfg).exists():
        print("Reading config.yaml")
        cfg = Config.load(inp_cfg)
    else:
        raise ValueError("No config.yaml")
        #cfg = Config()
    params = cfg.to_dict()

    enabled_detectors = ['hsd', 'ebeam', 'gmd', 'spect']
    assert len(enabled_detectors) > 0, "No detectors enabled!"
    # ['spect', 'ebeam', 'lcams', 'timing', 'xtcav']
    # note: rename vls <-> spect

    '''
    timings = []
    spects = []
    ebeams = []
    xtcavs = []

    # for i,run in enumerate(ds.runs()) hangs after last run
    ^C
        main(nshots,expname,runnums,scratchdir)
  File "/sdf/home/r/rogersdd/src/tmo-prefex/src/fex2h5.py", line 99, in main
    for i,run in enumerate(ds.runs()):
  File "/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_102424/psana/psana/psexp/serial_ds.py", line 89, in runs
    while self._start_run():
  File "/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_102424/psana/psana/psexp/serial_ds.py", line 79, in _start_run
    if self._setup_beginruns():  # try to get next run from current files
  File "/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_102424/psana/psana/psexp/serial_ds.py", line 74, in _setup_beginruns
    dgrams = self.smdr_man.get_next_dgrams()
  File "/sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_102424/psana/psana/psexp/smdreader_manager.py", line 147, in get_next_dgrams
    self.smdr.find_view_offsets(batch_size=1, ignore_transition=False)
KeyboardInterrupt
    '''


    ###################################
    #### Setting up the datasource ####
    ###################################
    ds = psana.DataSource(exp=expname,run=runnums)
    for i in range(len(runnums)):
        run = next(ds.runs()) # don't call next unless you know it's there...
        outname = '%s/hits.%s.run_%03i'%(scratchdir,expname,run.runnum)
        # 1. Setup detector configs (determining runs, saves)
        runs = []
        saves = []
        #fake_save = lambda lst: {} ## fake save for testing
        for detector_type in enabled_detectors:
            setup, get, save = detector_configs[detector_type]
            detectors = setup(run, params)
            runs.append(get(detectors, params))
            saves.append(save)

        # Save these params.
        if i == 0:
            print(f"Saving config to {cfgname}")
            Config.from_dict(params).save(cfgname, overwrite=True)

        # geometric sequence (1, 2, 4, ..., 512, 1024, 1024, ...)
        sizes = gseq(2) >> takewhile(lambda x: x<1024)
        sizes << seq(1024, 0)
        # 2. Assemble the stream to execute

        # - Start from a stream of (eventnum, event).
        s = run_events(run)
        if nshots > 0: # truncate to nshots?
            s >>= take(nshots)
        # - Run those through both run_hsds and run_gmds,
        #   producing a stream of ( Dict[DetectorID,HsdData],
        #                           Dict[DetectorID,GmdData] )
        s >>= split(*runs)
        # - but don't pass items that contain any None-s.
        #   (note: classes test as True)
        s >>= live_events()
        s >>= map(process_all)
        # - Now chop the stream into lists of length 128.
        s >>= chop(128)
        # - Now save each grouping as a "Batch".
        s >>= xmap(batch_data, saves)
        # - Now group those by increasing size and concatenate them.
        # - This makes increasingly large groupings of the output data.
        #s >>= chopper([1, 10, 100, 1000]) >> map(concat_batch) # TODO

        # 3. call send_hdf on ea. element passed
        #s >> tap(send_hdf)
        # 4. Further combine these into larger and larger
        #    chunk sizes for saving.
        s >>= variable_chunks(sizes) >> map(Batch.concat)

        # 5. The entire stream "runs" when connected to a sink:
        s >> write_out(outname)


    print("Hello, I'm done now.  Have a most excellent day!")

if __name__ == '__main__':
    if len(sys.argv)>3:
        nshots = int(sys.argv[1])
        expname = sys.argv[2]
        runnums = [int(r) for r in list(sys.argv[3:])]
        print('Before finalizing, clean up to point to common area for output .h5')
        scratchdir = '/sdf/data/lcls/ds/tmo/%s/scratch/%s/h5files'%(expname,os.environ.get('USER'))
        if not os.path.exists(scratchdir):
            os.makedirs(scratchdir)
        main(nshots,expname,runnums,scratchdir)
    else:
        print('Please give me a number of shots (-1 for all), experiment name, a list of run numbers, and output directory defaults to expt scratch directory)')
