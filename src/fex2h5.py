#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import Type,List,Dict
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
    take, Source, takei, seq
)

from new_port import PortConfig, WaveData, FexData
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
rungmd=False

def setup_hsd_ports(run, params):
    is_fex = params['is_fex']
    t0s = params['t0s']
    logicthresh = params['logicthresh']
    offsets = params['offsets']

    nr_expand = params['expand']
    inflate = params['inflate']

    rkey = run.runnum
    detslist = [s for s in run.detnames]
    hsdnames = [s for s in detslist if 'hsd' in s]

    port = {}
    hsds = {}
    chankeys = {}
    for hsdname in hsdnames:
        port.update({hsdname:{}})
        chankeys.update({hsdname:{}})
        if runhsd:
            assert hsdname in detslist
            hsd = run.Detector(hsdname)
            hsds[hsdname] = hsd
            for i,k in enumerate(list(hsd.raw._seg_configs().keys())):
                chankeys[hsdname].update({k:k}) # this we may want to replace with the PCIe address id or the HSD serial number.
                #print(f'{k} ~> {chankeys[hsdname][k]}')
                port[hsdname][k] = PortConfig(
                    id = k,
                    chankey = chankeys[hsdname][k],
                    is_fex = is_fex,
                    hsdname = hsdname,
                    inflate = inflate,
                    expand = nr_expand,
                    logic_thresh = 1<<12,
                    roll_on = 1<<6,
                    roll_off = 1<<6,
                #   t0=t0s[i]
                #   logicthresh=logicthresh[i]
                )
                #port[hsdname].update({k:Port(k,chankeys[hsdname][k],inflate=inflate,expand=nr_expand)})
                #port[hsdname][k].set_runkey(rkey).set_hsdname(hsdname)
                #port[hsdname][k].set_logicthresh(1<<12)
                if is_fex:
                    #port[hsdname][k].setRollOn((3*int(hsd.raw._seg_configs()[k].config.user.fex.xpre))>>2) # guessing that 3/4 of the pre and post extension for threshold crossing in fex is a good range for the roll on and off of the signal
                    #port[hsdname][k].setRollOff((3*int(hsd.raw._seg_configs()[k].config.user.fex.xpost))>>2)
                    port[hsdname][k].roll_on = (3*int(hsd.raw._seg_configs()[k].config.user.fex.xpre))>>2
                    port[hsdname][k].roll_off = (3*int(hsd.raw._seg_configs()[k].config.user.fex.xpost))>>2
                #else:
                #    port[hsdname][k].setRollOn(1<<6) 
                #    port[hsdname][k].setRollOff(1<<6)
    return port, hsds, chankeys

@source
def run_events(run, is_fex, port, hsds, chankeys, start_event=0):
    rkey = run.runnum
    print('starting analysis exp %s for run %i'%(expname,int(rkey)))

    '''
    print('processing run %i'%rkey)
    if runfzp and 'tmo_fzppiranha' in run.detnames:
        fzps += [run.Detector('tmo_fzppiranha')]
    else:
        runfzp = False

    if runtiming and '' in runs[r].detnames:
        timings += [runs[r].Detector('timing')]
    else:
        runtiming = False
    '''
    #wv = {}
    #wv_logic = {}
    #vsize = 0


    eventnum = start_event # later move this to outside the runs loop and let eventnum increase over all of the serial runs.

    for evt in run.events():
        out = {}
        completeEvent:List[bool] = [True]
        ## test hsds
        if runhsd and all(completeEvent):
            for hsdname, hsd in hsds.items():
                out[hsdname] = {}
                if hsd is None: # guard pattern
                    print(eventnum, f'hsd {hsdname} is None')
                    completeEvent += [False]
                    continue
                for key in chankeys[hsdname]: # here key means 'port number'
                    if is_fex:
                        peak = hsd.raw.peaks(evt)
                        if peak is None:
                            print('%i/%i: hsds[%s].raw.peaks(evt) is None'%(rkey,eventnum,hsdname))
                            completeEvent += [False]
                        else:
                            out[key] = FexData(port[hsdname][key],
                                                peak=peak)
                            completeEvent += [out[key].ok]
                    else:
                        wave = hsd.raw.waveforms(evt)
                        if wave is None:
                            out[key] = WaveData(port[hsdname][key],
                                                wave=wave)
                            completeEvent += [out[key].ok]
                        else:
                            print('%i/%i: hsds[%s].raw.waveforms(evt) is None'%(rkey,eventnum,hsdname))
                            completeEvent += [False]

        ## finish testing all detectors to measure ##
        ## before processing ##

        ## process hsds
        if runhsd and all(completeEvent):
            for hsdname, out_hsd in out.items():
                ''' HSD-Abaco section '''
                for key, data in out_hsd.items():
                    nwins:int = 1
                    xlist:List[int] = []
                    slist:List[ np.ndarray ] = []
                    baseline = np.uint32(0)
                    if data.cfg.is_fex:
                        nwins = len(data.peak[0])
                        if nwins >2 : # always reports the start of and the end of the fex active window.
                            baseline = np.sum(data.peak[1][0].astype(np.uint32))
                            baseline //= len(data.peak[1][0])
                            xlist.extend(data.peak[0][1:nwins-2])
                            for i in range(1,nwins-2):
                                #xlist += [ data.peak[0][i] ]
                                slist += [ np.array(data.peak[1][i],dtype=np.int32) ]
                    else:
                        slist += [ np.array(data.wave, dtype=np.int16) ] # presumably 12 bits unsigned input, cast as int16_t since will immediately in-place subtract baseline
                        xlist += [0]
                    data.set_baseline(baseline)
                    #.process(slist,xlist) # this making a list out of the waveforms is to accommodate both the fex and the non-fex with the same Port object and .process() method.
        # Note: We yield once here for every event
        # and keep hsdEvents in a separate (accumulator) function

        ## redundant events vec
        # NOTE: eventnum only increments for complete events
        if all(completeEvent):
            yield eventnum, out
            eventnum += 1

@stream
def accumEvents(inp, hsds, chankeys):
    """ Accumulate a growing list of complete events.
    """
    hsdEvents = []
    #if runhsd: # assume this is true because you called me

    init = True

    for eventnum, port in inp:
        hsdEvents.append(eventnum)

        if init:
            init = False
            for hsdname in port.keys():
                for key in port[hsdname].keys():
                    port[hsdname][key].set_initState(False)

        if eventnum<2:
            for hsdname in hsds.keys():
                print('ports = %s'%([k for k in chankeys[hsdname].keys()]))
        yield eventnum, port, hsdEvents

# inp: Iterator[Tuple[ int, Dict[str,Dict[int,Port]], List[int] ]]
@sink
def write_out(inp, outname):
    #if runhsd: # assume this is true because you called me
    for (eventnum, port, hsdEvents) in inp:
        for hsdname in hsds.keys():
            print('%s: writing event %i,\tnedges = %s'%(hsdname, eventnum,[port[hsdname][k].getnedges() for k in chankeys[hsdname]] ))

        print('writing to %s'%outname)
        with h5py.File(outname,'w') as f:
            Port.update_h5(f,port,hsdEvents)

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
    gmd = [Gmd() for r in runnums]
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
    is_fex = params['is_fex']
    
    selected = Source(range(100, 1000, 100))
    selected << seq(1000, 1000)

    ds = psana.DataSource(exp=expname,run=runnums)
    for r in runnums:
        run = next(ds.runs())
        outname = '%s/hits.%s.run_%03i.h5'%(scratchdir,expname,run.runnum)
        port, hsds, chankeys = setup_hsd_ports(run, params)
        s = run_events(run, is_fex, port, hsds, chankeys)
        if nshots > 0: # truncate to nshots
            s = s >> take(nshots)
        s >> accumEvents(hsds, chankeys) \
          >> takei(selected) \
          >> write_out(outname)

    print("Hello, I'm done now.  Have a most excellent day!")


if __name__ == '__main__':
    if len(sys.argv)>3:
        nshots = int(sys.argv[1])
        expname = sys.argv[2]
        runnums = [int(r) for r in list(sys.argv[3:])]
        scratchdir = '/sdf/data/lcls/ds/tmo/%s/scratch/%s/h5files/%s'%(expname,os.environ.get('USER'),socket.gethostname())
        if not os.path.exists(scratchdir):
            os.makedirs(scratchdir)
        main(nshots,expname,runnums,scratchdir)
    else:
        print('Please give me a number of shots (-1 for all), experiment, a list of run numbers, and output directory is to expt scratch)')
