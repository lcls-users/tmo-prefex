from typing import List,Dict,Optional,Tuple,Union,Any
from typing_extensions import Annotated

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
def accum_out(inp: Iterator[Batch], outdir: Path,
              fprefix: str, stepinfo: Dict[str,Any]
             ) -> Iterator[Batch]:
    """Accumulate all events into one giant file.
       Write accumulated result with each iterate.

       Yields only the input batch for each iteration.
    """
    batch0 = None
    for i, batch in enumerate(inp):
        if batch0 is None:
            batch0 = batch
        else:
            batch0.extend(batch)
        # Extend instead of writing separate files.
        outz = outdir/f'{fprefix}.h5'
        print('appending batch %i to %s'%(i,outz))
        with h5py.File(outz,'w') as h:
            for k, v in stepinfo.items():
                h.attrs.create(k, data=v)
            batch0.write_h5(h)

        yield batch

@stream
def write_out(inp: Iterator[Batch], outdir: Path,
              fprefix: str, stepinfo: Dict[str,Any]
             ) -> Iterator[Batch]:
    """Write each batch of events to its own h5 file.
    TODO: accumulate events until batch.size() passes
    a threshold.
    """
    for i, batch in enumerate(inp):
        outz = outdir/f'{fprefix}.{i:03d}.h5'
        print('writing batch %i to %s'%(i,outz))
        with h5py.File(outz,'w') as h:
            for k, v in stepinfo.items():
                h.attrs.create(k, data=v)
            batch.write_h5(h)

        yield batch
        #yield batch.attrs['events']

def serialize_h5(batch: Batch,
                 stepinfo: Dict[str,Any]) -> bytes:
    with io.BytesIO() as f:
        with h5py.File(f, 'w') as h:
            for k, v in stepinfo.items():
                h.attrs.create(k, data=v)
            batch.write_h5(h)
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
def main(expname: Annotated[
                    str,
                    typer.Argument(help="Experiment name"),
         ],
         run: Annotated[
                    int,
                    typer.Argument(help="Run number"),
         ],
         detectors: Annotated[
                    str,
                    typer.Argument(help="Comma-separated list of detectors (e.g. gmd,hsd,spect)"),
         ],
         config: Annotated[
                    Optional[Path],
                    typer.Option(help="Detector configuration file"),
         ] = None,
         outdir: Annotated[
                    Optional[Path],
                    typer.Option(
                        rich_help_panel="Output prefix for hdf5 files",
                        help="Defaults to /sdf/scratch/lcls/ds/{abbr}/{expname}/scratch/xtc2h5/run_{run:03d}/{cfg_hash}")
         ] = None,
         dial: Annotated[
                    Optional[str],
                    typer.Option(help="Detector configuration file"),
         ] = None,
        ):

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
            print("Warning: no config.yaml provided -- all detector settings at default.")
        cfg = Config() # empty
    elif config.exists():
        if rank == 0:
            print(f"Reading config from {config}")
        cfg = Config.load(config)
    else:
        raise RuntimeError(f"Unable to read {config}")
    params = cfg.to_dict()

    # Set config-specific output directory.
    if outdir is None:
        abbr = expname[:3]
        cfg_hash = cfg.hash(8)
        outdir = Path(f'/sdf/scratch/lcls/ds/{abbr}/{expname}/scratch/xtc2h5/run_{run:03d}/{cfg_hash}')
    outdir.mkdir(exist_ok=True, parents=True)

    enabled_detectors = detectors.split(',')
    assert len(enabled_detectors) > 0, "No detectors enabled!"
    for det in enabled_detectors:
        if det not in detector_configs:
            raise KeyError(f"Unknown detector type: {det}")

    ###################################
    #### Setting up the datasource ####
    ###################################
    runnums = [run]
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
            cfgname = outdir / "config.yaml"
            print(f"Saving config to {cfgname}")
            Config.from_dict(params).save(cfgname, overwrite=True)

        t0 = run.timestamp
        for sid, step in enumerate(run.steps()):
            stepname = f"step_{sid+1:02d}"
            stepinfo = {'run': runnum, 'step_value': sid+1}
            for it in run.scaninfo.items():
                name = it[0][0]
                v = run.Detector(name)(step)
                stepinfo[name] = v
                if name == 'step_value':
                    # If present, it should be defined for all steps.
                    v = int(v)
                    stepname = f"step_{v:02d}"

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

            # Output names will be 'step_MM[-rank].JJJ.h5'
            #                      = 'stepname+rank_suf.JJJ.h5'
            # * The entire stream "runs" when connected to a sink:
            s >>= write_out(outdir, stepname+rank_suf, stepinfo)
            if dial is not None:
                send_pipe = xmap(serialize_h5, stepname+rank_suf, stepinfo) \
                            >> pusher(dial, 1)
                s >>= split(map(lambda b: b), send_pipe) >> cut[0]
            for stat in s >> map(lambda b: b.attrs['events']) \
                          >> clock():
                # Create statistics on events processed and time
                # per iteration and print all those out.
                print(stat, flush=True)

    if rank == 0: # Write a sentinel file indicating data is complete.
        (outdir/"done").write_text("processed all events\n")
        print("Hello, I'm done now.  Have a most excellent day!")
