#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import List,Dict,Optional,Tuple,Union
from collections.abc import Iterator
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

from Hsd import HsdConfig, WaveData, FexData, save_hsd
from Ebeam import *
#from Vls import *
from Gmd import GmdConfig, GmdData, save_gmd
from Spect import *
from Config import Config
#from utils import *
from stream_utils import split, xmap

from combine import batch_data, Batch

# Some types:
DetectorID   = Tuple[str, int] # ('hsd', 22)
DetectorData = Union[WaveData, FexData, GmdData]
EventData    = Dict[DetectorID, DetectorData]

def setup_gmds(run, params):
    # gmd-s store their output in xray-s
    gmdnames = [s for s in run.detnames if s.endswith("gmd")]
    
    gmds = {}
    xray = {}
    for gmdname in gmdnames:
        gmd = run.Detector(gmdname)
        if gmd is None:
            print(f'run.Detector({gmdname}) is None!')
            continue
        gmds[(gmdname,0)] = gmd

        cfg = GmdConfig(
            name = gmdname
        )
        if 'x' in gmdname: # or gmdname.endswith('x')
            cfg.unit = '0.1uJ'
            cfg.scale = 10000

        xray[(gmdname,0)] = cfg
    return gmds, xray

def setup_hsds(run, params):
    # hsd-s store their output in "port"-s
    is_fex = params['is_fex']
    t0s = params['t0s']
    logicthresh = params['logicthresh']
    offsets = params['offsets']

    nr_expand = params['expand']
    inflate = params['inflate']

    hsdnames = [s for s in run.detnames if s.endswith('hsd')]

    hsds = {}
    ports = {}
    for hsdname in hsdnames:
        hsd = run.Detector(hsdname)
        if hsd is None:
            print(f'run.Detector({hsdname}) is None!')
            continue

        hsds[hsdname] = hsd
        # FIXME: do we need .raw here?
        # can the keys() be ordered into a contiguous range?
        for i,k in enumerate(list(hsd.raw._seg_configs().keys())):
            ports[(hsdname,k)] = HsdConfig(
                id = k,
                # may want to use chankey to store the PCIe address id
                # or the HSD serial number.
                chankey = k,
                is_fex = is_fex,
                name = hsdname,
                inflate = inflate,
                expand = nr_expand,
                logic_thresh = 18000,
                roll_on = 1<<6,
                roll_off = 1<<6,
            #   t0=t0s[i]
            #   logicthresh=logicthresh[i]
            )
            if is_fex:
                # guessing that 3/4 of the pre and post extension for
                # threshold crossing in fex is a good range for the
                # roll on and off of the signal
                ports[(hsdname,k)].roll_on = (3*int(hsd.raw._seg_configs()[k].config.user.fex.xpre))>>2
                ports[(hsdname,k)].roll_off = (3*int(hsd.raw._seg_configs()[k].config.user.fex.xpost))>>2
    return hsds, ports

@source
def run_events(run, start_event=0):
    for i, evt in enumerate(run.events()):
        yield i+start_event, evt

@stream
def run_gmds(events, gmds, xray,
            ) -> Iterator[Optional[EventData]]:
    # assume rungmd is True if this fn. is called

    for eventnum, evt in events:
        completeEvent = True

        out = {}
        for idx, gmd in gmds.items():
            out[idx] = GmdData(xray[idx],
                              eventnum,
                              gmd.raw.milliJoulesPerPulse(evt))
            completeEvent = out[idx].ok
            if not completeEvent:
                break

        if completeEvent:
            yield out
        else:
            yield None

@stream
def run_hsds(events, hsds, ports,
            ) -> Iterator[Optional[EventData]]:
    # assume runhsd is True if this fn. is called

    for eventnum, evt in events:
        out = {}

        ## test hsds
        completeEvent = True
        for (hsdname, key), port in ports.items():
            # here key means 'port number'
            hsd = hsds[hsdname]

            idx = (hsdname, key)
            if port.is_fex:
                peak = hsd.raw.peaks(evt)
                if peak is None:
                    print('%i: hsds[%s].raw.peaks(evt) is None'%(eventnum,repr(idx)))
                    completeEvent = False
                else:
                    out[idx] = FexData(port,
                                       eventnum,
                                       peak=peak[key][0])
                    completeEvent = out[idx].ok
            else:
                wave = hsd.raw.waveforms(evt)
                if wave is None:
                    print('%i: hsds[%s].raw.waveforms(evt) is None'%(eventnum,repr(idx)))
                    completeEvent = False
                else:
                    out[idx] = WaveData(port,
                                        eventnum,
                                        wave=wave[key][0])
                    completeEvent = out[idx].ok
            if not completeEvent:
                break

        ## finish testing all detectors to measure ##
        ## before processing ##

        ## process hsds
        for idx, data in out.items():
            ''' HSD-Abaco section '''
            if not data.setup().process():
                completeEvent = False
                break

        if completeEvent:
            yield out
        else:
            yield None

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
    cfgname = '%s/%s.%s.configs.h5'%(scratchdir,expname,os.environ.get('USER'))
    configs = Config(is_fex=True)
    params = configs.writeconfigs(cfgname).getparams()

    '''
    spect = [Vls(params['vlsthresh']) for r in runnums]
    _ = [s.setwin(params['vlswin'][0],params['vlswin'][1]) for s in spect]
    #_ = [s.setthresh(params['vlsthresh']) for s in spect]
    ebunch = [Ebeam() for r in runnums]
    _ = [e.setoffset(params['l3offset']) for e in ebunch]
    '''
    runhsd=True
    rungmd=False
    runlcams=False
    runtiming=False

    runvls=False
    runebeam=False
    runxtcav=False

    '''
    timings = []
    vlss = []
    ebeams = []
    xtcavs = []
    '''

    ###################################
    #### Setting up the datasource ####
    ###################################
    ds = psana.DataSource(exp=expname,run=runnums)
    for r in runnums:
        run = next(ds.runs())
        outname = '%s/hits.%s.run_%03i.h5'%(scratchdir,expname,run.runnum)
        hsds, ports = setup_hsds(run, params)
        gmds,  xray = setup_gmds(run, params)

        print('ports: ', list(ports.keys()))

        # Start from a stream of (eventnum, event).
        s = run_events(run)
        # Run those through both run_hsds and run_gmds,
        # producing a stream of ( Dict[DetectorID,HsdData],
        #                         Dict[DetectorID,GmdData] )
        s >>= split(run_hsds(hsds, ports), run_gmds(gmds, xray))
        # but don't pass items that contain any None-s.
        # (note: classes test as True)
        s >>= filter(all)
        if nshots > 0: # truncate to nshots
            s >>= take(nshots)
        # Now chop the stream into lists of length 100.
        s >>= chop(100)
        # Now save each grouping as a "Batch".
        #fake_save = lambda lst: {}
        #s >>= xmap(batch_data, [save_hsd, fake_save])
        s >>= xmap(batch_data, [save_hsd, save_gmd])
        # Now group those by increasing size and concatenate them.
        # This makes increasingly large groupings of the output data.
        #s >>= chopper([1, 10, 100, 1000]) >> map(concat_batch) # TODO

        # The entire stream "runs" when connected to a sink:
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
