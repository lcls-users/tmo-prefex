#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import List,Dict,Optional,Tuple,Union
from collections.abc import Iterator
from pathlib import Path
import sys
import re
import os
import socket

import h5py
import psana
import numpy as np
# python3 -m venv --system-site-packages ./venv
# . ./venv/bin/activate
# which pip # ensure pip path points to venv!!!
# pip install -r ../requirements.txt
from stream import (
    stream, source, sink,
    take, Source, takei, seq,
    chop, map, filter
)

from Config import Config

from Hsd import HsdConfig, WaveData, FexData, setup_hsds, save_hsd
from Ebeam import EbeamConfig, EbeamData, setup_ebeams, save_ebeam
from Gmd import GmdConfig, GmdData, setup_gmds, save_gmd
from Spect import SpectConfig, SpectData, setup_spects, save_spect

# Some types:
DetectorID   = Tuple[str, int] # ('hsd', 22)
DetectorData = Union[WaveData, FexData, GmdData, EbeamData, SpectData]
EventData    = Dict[DetectorID, DetectorData]

def save_fex(run, params):
    return setup_hsds(run, params, default_fex)

# Plugin system for detector types:
detector_configs = {
    'hsd': (setup_fex, run_hsds, save_hsd),
    'ebeam': (setup_ebeams, run_ebeams, save_ebeam),
    'gmd': (setup_gmds, run_gmds, save_gmd),
    'spect': (setup_spects, run_spects, save_spect),
}

from stream_utils import split, xmap
from combine import batch_data, Batch

default_wave = HsdConfig(
    id=0, # psana's name for detector
    chankey=0, # our name for detector
    # -- may want to use chankey to store the PCIe
    #    address id or the HSD serial number.
    is_fex = False,
    name = '',
    inflate = 2,
    expand = 4,
    logic_thresh = 18000,
    roll_on = 1<<6,
    roll_off = 1<<6,
)
default_fex = HsdConfig(
    id=0, # psana's name for detector
    chankey=0, # our name for detector
    # -- may want to use chankey to store the PCIe
    #    address id or the HSD serial number.
    is_fex = True,
    name = '',
    inflate = 2,
    expand = 4,
    logic_thresh = 18000,
)
#   t0=t0s[i]
#   logicthresh=logicthresh[i]

@source
def run_events(run, start_event=0):
    for i, evt in enumerate(run.events()):
        yield i+start_event, evt

@sink
def write_out(inp: Iterator[Batch], outname: str) -> None:
    for i, batch in enumerate(inp):
        name = outname[:-2] + f"{i}.h5"
        print('writing batch %i to %s'%(i,name))
        with h5py.File(name,'w') as f:
            batch.write_h5(f)

def main(nshots:int, expname:str, runnums:List[int], scratchdir:str):
    #######################
    #### CONFIGURATION ####
    #######################
    cfgname = '%s/%s.%s.configs.yaml'%(scratchdir,expname,os.environ.get('USER'))
    inp_cfg = 'config.yaml'
    if Path(inp_cfg).exists():
        cfg = Config.load(inp_cfg)
    else:
        cfg = Config()
    params = cfg.to_dict()

    enabled_detectors = ['hsd', 'gmd']
    assert len(enabled_detectors) > 0, "No detectors enabled!"
    # ['spect', 'ebeam', 'lcams', 'timing', 'xtcav']
    # note: rename vls <-> spect

    '''
    timings = []
    spects = []
    ebeams = []
    xtcavs = []
    '''

    ###################################
    #### Setting up the datasource ####
    ###################################
    ds = psana.DataSource(exp=expname,run=runnums)
    for i,run in enumerate(ds.runs()):
        outname = '%s/hits.%s.run_%03i.h5'%(scratchdir,expname,run.runnum)
        # 1. Setup detector configs (determining runs, saves)
        runs = []
        saves = []
        #fake_save = lambda lst: {} ## fake save for testing
        for detector_type in enabled_detectors:
            setup, run, save = detector_configs[detector_type]
            detectors = setup(run, params)
            runs.append(run(detectors, params))
            saves.append(save)

        # Save these params.
        if i == 0:
            Config.from_dict(params).save(cfgname)
            print(f"Saving config to {cfgname}")

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
        s >>= filter(all)
        # - Now chop the stream into lists of length 100.
        s >>= chop(100)
        # - Now save each grouping as a "Batch".
        s >>= xmap(batch_data, saves)
        # - Now group those by increasing size and concatenate them.
        # - This makes increasingly large groupings of the output data.
        #s >>= chopper([1, 10, 100, 1000]) >> map(concat_batch) # TODO

        # 3. The entire stream "runs" when connected to a sink:
        s >> write_out(outname)


    print("Hello, I'm done now.  Have a most excellent day!")

if __name__ == '__main__':
    if len(sys.argv)>3:
        nshots = int(sys.argv[1])
        expname = sys.argv[2]
        runnums = [int(r) for r in list(sys.argv[3:])]
        print('Before finalizing, clean up to point to common area for output .h5')
        scratchdir = '/sdf/data/lcls/ds/tmo/%s/scratch/%s/h5files/%s'%(expname,os.environ.get('USER'),socket.gethostname())
        if not os.path.exists(scratchdir):
            os.makedirs(scratchdir)
        main(nshots,expname,runnums,scratchdir)
    else:
        print('Please give me a number of shots (-1 for all), experiment name, a list of run numbers, and output directory defaults to expt scratch directory)')
