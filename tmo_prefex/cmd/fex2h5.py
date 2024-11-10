#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import List,Dict,Optional,Tuple,Union,Any
from collections.abc import Iterator
from pathlib import Path
import sys
import re
import os
import io

import typer
import h5py
import psana
import numpy as np
from stream import (
    stream, source, sink,
    take, Source, takei, seq, gseq,
    chop, map, filter, takewhile, cut
)

from lclstream.nng import pusher
from lclstream.stream_utils import clock

from ..Config import Config
from ..stream_utils import variable_chunks, split, xmap

from ..Hsd import HsdConfig, WaveData, FexData, run_hsds, setup_hsds, save_hsd
from ..Ebeam import EbeamConfig, EbeamData, setup_ebeams, run_ebeams, save_ebeam
from ..Gmd import GmdConfig, GmdData, setup_gmds, run_gmds, save_gmd
from ..Spect import SpectConfig, SpectData, setup_spects, run_spects, save_spect
from ..combine import batch_data, Batch

# Some types:
DetectorID   = Tuple[str, int] # ('hsd', 22)
DetectorData = Union[WaveData, FexData, GmdData, EbeamData, SpectData]
EventData    = Dict[DetectorID, DetectorData]

app = typer.Typer(pretty_exceptions_enable=False)

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

@source
def run_events(run, t0):
    # Note: using timesamp_diff instead of sequence number
    # because it maintains uniqueness if fex2h5 is run in parallel.
    #t0 = run.timestamp

    # note, that these numbers are O(112_538_353)
    # and typically step by 120_000
    # maybe nanoseconds units?
    for evt in run.events():
        t = evt.timestamp_diff(t0)
        yield (t+500)//1000, evt

@stream
def write_out(inp: Iterator[Batch], outname: str,
              stepname: str, stepinfo: Dict[str,Any]) -> Iterator[int]:
    """Accumulate all events into one giant file.
       Write accumulated result with each iterate.
    """
    batch0 = None
    for i, batch in enumerate(inp):
        if batch0 is None:
            batch0 = batch
        else:
            batch0.extend(batch)
        # Extend instead of writing separate files.
        print('writing batch %i to %s'%(i,outname))
        with h5py.File(outname,'w') as f:
            g = f.create_group(stepname)
            for k, v in stepinfo.items():
                g.attrs.create(k, data=v)
            batch0.write_h5(g)

        try:
            nev = len(batch[next(batch.keys())].events)
        except Exception:
            nev = 0
        yield nev

def serialize_h5(batch: Batch,
                 stepname: str,
                 stepinfo: Dict[str,Any]) -> bytes:
    with io.BytesIO() as f:
        with h5py.File(f, 'w') as h:
            g = f.create_group(stepname)
            for k, v in stepinfo.items():
                g.attrs.create(k, data=v)
            batch.write_h5(g)
        return f.getvalue()

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
    ev = 0
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

def mk_sizes(upto=1024):
    # geometric sequence (1, 2, 4, ..., 512, 1024, 1024, ...)
    sizes = gseq(2) >> takewhile(lambda x: x<upto)
    sizes << seq(upto, 0)
    return sizes

@app.command()
def main(nshots: int, expname: str,
         runnums: List[int],
         dial: Optional[str] = None,
         scratchdir: Optional[str] = None):
    if scratchdir is None:
        user = os.environ.get('USER')
        scratchdir = '/sdf/data/lcls/ds/tmo/%s/scratch/%s/h5files'%(expname,user)
    if not os.path.exists(scratchdir):
        os.makedirs(scratchdir)

    # Check whether MPI is enabled, and adjust output filename
    # appropriately
    rank = psana.MPI.COMM_WORLD.Get_rank()
    ranks = psana.MPI.COMM_WORLD.Get_size()
    rank_suf = ''
    if ranks > 1:
        rank_suf = f'-{rank:03d}'

    #######################
    #### CONFIGURATION ####
    #######################
    cfgname = '%s/%s.%s.configs.yaml'%(scratchdir,expname,os.environ.get('USER'))
    inp_cfg = 'config.yaml'
    if Path(inp_cfg).exists():
        if rank == 0:
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
        if i == 0 and rank == 0:
            print(f"Saving config to {cfgname}")
            Config.from_dict(params).save(cfgname, overwrite=True)

        t0 = run.timestamp
        for sid, step in enumerate(run.steps()):
            stepname = f"step_{sid+1:02d}"
            stepinfo = {}
            for it in run.scaninfo.items():
                name = it[0][0]
                v = run.Detector(name)(step)
                if name == 'step_value':
                    # If present, it should be defined for all steps.
                    v = int(v)
                    stepname = f"step_{v:02d}"
                else:
                    stepinfo[name] = v

            # * Assemble the stream to execute
            #   - Start from a stream of (eventnum, event).
            s = run_events(step, t0)

            if nshots > 0: # truncate to nshots?
                s >>= take(nshots)
            #   - Run those through both run_hsds and run_gmds,
            #     producing a stream of ( Dict[DetectorID,HsdData],
            #                             Dict[DetectorID,GmdData] )
            s >>= split(*runs)
            #   - but don't pass items that contain any None-s.
            #     (note: classes test as True)
            s >>= live_events()
            s >>= map(process_all)
            #   - chop the stream into lists of length n.
            s >>= chop(512)
            #   - save each grouping as a "Batch".
            s >>= xmap(batch_data, saves)
            #   - Further combine these into larger and larger
            #     chunk sizes for saving.
            s >>= variable_chunks(mk_sizes()) >> map(Batch.concat)

            outname = '%s/hits.%s.run_%03i.%s%s.h5'%(
                        scratchdir, expname, run.runnum,
                        stepname, rank_suf)
            # * The entire stream "runs" when connected to a sink:
            if dial is None:
                s >>= write_out(outname, stepname, stepinfo)
            else:
                send_pipe = xmap(serialize_h5, stepname, stepinfo) \
                            >> pusher(dial, 1)
                # do both hdf5 file writing
                # and send_pipe.  Use cut[0] to pass only
                # the result of write_out (event counts).
                s >>= split(write_out(outname, stepname, stepinfo),
                            send_pipe) \
                      >> cut[0]
            for stat in s >> clock():
                print(stat, flush=True)

    print("Hello, I'm done now.  Have a most excellent day!")
