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
from ..detectors import detector_configs

from ..combine import batch_data, Batch

app = typer.Typer(pretty_exceptions_enable=False)

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
def accum_out(inp: Iterator[Batch], outname: str,
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

@stream
def write_out(inp: Iterator[Batch], outname: str,
              stepname: str, stepinfo: Dict[str,Any]) -> Iterator[int]:
    """Write each batch of events to its own h5 file.
    TODO: accumulate events until batch.size() passes
    a threshold.
    """
    for i, batch in enumerate(inp):
        outz = f'{outname[:-3]}.{i:03d}.h5'
        print('writing batch %i to %s'%(i,outz))
        with h5py.File(outz,'w') as f:
            g = f.create_group(stepname)
            for k, v in stepinfo.items():
                g.attrs.create(k, data=v)
            batch.write_h5(g)

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
def main(expname: str, detectors: str,
         runnums: List[int],
         config: Optional[Path] = None,
         dial: Optional[str] = None,
         outdir: Optional[Path] = None):

    nshots = 0 # don't truncate

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
    if config is None:
        if rank == 0:
            print("Warning: empty config.yaml -- all detector settings at default.")
        cfg = Config() # empty
    elif config.exists():
        if rank == 0:
            print("Reading config.yaml")
        cfg = Config.load(config)
    else:
        raise RuntimeError(f"Unable to read {config}")
    params = cfg.to_dict()

    # Set config-specific output directory.
    if outdir is None:
        user = os.environ.get('USER')
        outdir = Path('/sdf/data/lcls/ds/tmo/%s/scratch/%s/psana2h5'%(expname,user))
    outdir = outdir / cfg.hash(8)
    outdir.mkdir(exist_ok=True, parents=True)

    enabled_detectors = detectors.split(',')
    assert len(enabled_detectors) > 0, "No detectors enabled!"
    for det in enabled_detectors:
        if det not in detector_configs:
            raise KeyError(f"Unknown detector type: {det}")

    ###################################
    #### Setting up the datasource ####
    ###################################
    ds = psana.DataSource(exp=expname,run=runnums)
    for i, runnum in enumerate(runnums):
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
            stepinfo = {'run': runnum}
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
            s >>= chop(1024)
            #   - save each grouping as a "Batch".
            s >>= xmap(batch_data, saves)
            #   - Further combine these into larger and larger
            #     chunk sizes for saving.
            s >>= variable_chunks(mk_sizes()) >> map(Batch.concat)

            outname = outdir/'%s.run_%03i.%s%s.h5'%(
                                        expname, run.runnum,
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
