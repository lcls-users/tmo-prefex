#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import Type,List,Dict,Optional
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

from new_port import PortConfig, WaveData, FexData
from combine_port import save_dd_batch, Batch
from Ebeam import *
from Vls import *
from Gmd import *
from Config import Config
from utils import *

runhsd=True
runfzp=False
runtiming=False

runvls=False
runebeam=False
runxtcav=False
rungmd=True

def setup_gmd_xrays(run, params):
    # gmd-s store their output in "xray"-s
    gmdnames = [s for s in run.detnames if s.endswith("gmd")]
    
    gmds = {}
    xray = {}
    for gmdname in gmdnames:
        gmd[gmdname] = run.Detector(gmdname)

        cfg = XrayConfig(
            name = gmdname
        )
        if 'x' in gmdname:
            cfg.unit = '0.1uJ'
            cfg.scale = 1e4

        xray[gmdname] = cfg
    return gmds, xray

def setup_hsd_ports(run, params):
    # hsd-s store their output in "port"-s
    is_fex = params['is_fex']
    t0s = params['t0s']
    logicthresh = params['logicthresh']
    offsets = params['offsets']

    nr_expand = params['expand']
    inflate = params['inflate']

    hsdnames = [s for s in run.detnames if s.endswith('hsd')]

    port = {}
    hsds = {}
    for hsdname in hsdnames:
        hsd = run.Detector(hsdname)
        if hsd is None:
            print(f'run.Detector({hsdname}) is None!')
            continue

        hsds[hsdname] = hsd
        for i,k in enumerate(list(hsd.raw._seg_configs().keys())):
            port[(hsdname,k)] = PortConfig(
                id = k,
                # may want to use chankey to store the PCIe address id
                # or the HSD serial number.
                chankey = k,
                is_fex = is_fex,
                # TODO: change hsdname ~> name
                hsdname = hsdname,
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
                port[(hsdname,k)].roll_on = (3*int(hsd.raw._seg_configs()[k].config.user.fex.xpre))>>2
                port[(hsdname,k)].roll_off = (3*int(hsd.raw._seg_configs()[k].config.user.fex.xpost))>>2
    return port, hsds

EventData = Dict[str,Dict[int,Any]]

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
        for gmdname, gmd in gmds.items():
            out[gmdname] = GmdData(xray[gmdname],
                              eventnum,
                              gmd.raw.milliJoulesPerPulse(evt))
            completeEvent = out[gmdname].ok
            if not completeEvent:
                break

        if completeEvent:
            yield out
        else:
            yield None

@source
def run_hsds(events, ports, hsds,
            ) -> Iterator[Optional[EventData]]:
    # assume runhsd is True if this fn. is called

    eventnum = start_event # later move this to outside the runs loop and let eventnum increase over all of the serial runs.

    for evt in run.events():
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
                    print('%i: hsds[%s,%i].raw.peaks(evt) is None'%(eventnum,hsdname,key))
                    completeEvent = False
                else:
                    out[idx] = FexData(port,
                                       eventnum,
                                       peak=peak)
                    completeEvent = out[idx].ok
            else:
                wave = hsd.raw.waveforms(evt)
                if wave is None:
                    print('%i: hsds[%s,%i].raw.waveforms(evt) is None'%(eventnum,hsdname,key))
                    completeEvent = False
                else:
                    out[idx] = WaveData(ports[hsdname][key],
                                        eventnum,
                                        wave=wave)
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

        # Note: We yield once here for every event
        eventnum += 1
        if completeEvent:
            yield out
        else:
            yield None

@sink
def write_out(inp : Iterator[Batch], outname: str) -> None:
    for i, batch in enumerate(inp):
        #hsdname, hsd = next(batch.items())
        #print('%s: nedges = '%hsdname, {k:p['nedges'] for k,p in hsd.items()})

        name = outname[:-2] + f"{i}.h5"
        print('writing batch %i to %s'%(i,name))
        with h5py.File(name,'w') as f:
            batch.write_h5(f)

def multi_detector_events():
    pass

def main(nshots:int,expname:str,runnums:List[int],scratchdir:str):
  
    outnames = {}


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

    '''
    timings = []
    fzps = []
    vlss = []
    ebeams = []
    xtcavs = []
    xgmds = []
    '''

    ###################################
    #### Setting up the datasource ####
    ###################################
    ds = psana.DataSource(exp=expname,run=runnums)
    for r in runnums:
        run = next(ds.runs())
        outname = '%s/hits.%s.run_%03i.h5'%(scratchdir,expname,run.runnum)
        ports, hsds = setup_hsd_ports(run, params)
        gmds,  xray = setup_gmd_xrays(run, params)

        for hsdname, p in ports.items():
            print('%s: ports = %s'%(hsdname, list(p.keys())))

        s = run_events(run) >> (
            run_hsds(ports, hsds) & run_gmds(gmds, xray) ) \
            >> filter(all)
        if nshots > 0: # truncate to nshots
            s = s >> take(nshots)
        s = s >> chop(100) >> map(save_dd_batch)

        # executes when connected to sink:
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
