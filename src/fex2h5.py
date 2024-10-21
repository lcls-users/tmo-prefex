#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

import psana
import numpy as np
import sys
import re
import h5py
import os
import socket

from typing import Type,List

from Ports import *
from Ebeam import *
from Vls import *
from Gmd import *
from Config import Config
from utils import *




def main(nshots:int,expname:str,runnums:List[int],scratchdir:str):
  
    outnames = {}

    _=[print('starting analysis exp %s for run %i'%(expname,int(r))) for r in runnums]

    #######################
    #### CONFIGURATION ####
    #######################
    cfgname = '%s/%s.%s.configs.h5'%(scratchdir,expname,os.environ.get('USER'))
    configs = Config(is_fex=True)
    params = configs.writeconfigs(cfgname).getparams()
    is_fex = params['is_fex']
    t0s = params['t0s']
    logicthresh = params['logicthresh']
    offsets = params['offsets']

    nr_expand = params['expand']
    inflate = params['inflate']

    '''
    spect = [Vls(params['vlsthresh']) for r in runnums]
    _ = [s.setwin(params['vlswin'][0],params['vlswin'][1]) for s in spect]
    #_ = [s.setthresh(params['vlsthresh']) for s in spect]
    ebunch = [Ebeam() for r in runnums]
    gmd = [Gmd() for r in runnums]
    _ = [e.setoffset(params['l3offset']) for e in ebunch]
    '''
    runhsd=True
    runfzp=False
    runtiming=False

    runvls=False
    runebeam=False
    runxtcav=False
    rungmd=False

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
    
    port = {}
    chankeys = {}
    hsds = {}
    hsdstring = {}
    ds = psana.DataSource(exp=expname,run=runnums)
    detslist = {}
    hsdnames = {}
    for r in runnums:
        run = next(ds.runs())
        rkey = run.runnum
        port.update({rkey:{}})
        hsds.update({rkey:{}})
        chankeys.update({rkey:{}})
        detslist.update({rkey:[s for s in run.detnames]})
        outnames.update({rkey:'%s/hits.%s.run_%03i.h5'%(scratchdir,expname,rkey)})

        hsdslist = [s for s in detslist[rkey] if re.search('hsd',s)]

        hsdnames.update({rkey:hsdslist})

        print('writing to %s'%outnames[rkey])
        for hsdname in hsdnames[rkey]:
            port[rkey].update({hsdname:{}})
            chankeys[rkey].update({hsdname:{}})
            if runhsd and hsdname in detslist[rkey]:
                hsds[rkey].update({hsdname:run.Detector(hsdname)})
                port[rkey].update({hsdname:{}})
                chankeys[rkey].update({hsdname:{}})
                for i,k in enumerate(list(hsds[rkey][hsdname].raw._seg_configs().keys())):
                    chankeys[rkey][hsdname].update({k:k}) # this we may want to replace with the PCIe address id or the HSD serial number.
                    #print(k,chankeys[rkey][hsdname][k])
                    #port[rkey][hsdname].update({k:Port(k,chankeys[rkey][hsdname][k],t0=t0s[i],logicthresh=logicthresh[i],inflate=inflate,expand=nr_expand)})
                    port[rkey][hsdname].update({k:Port(k,chankeys[rkey][hsdname][k],inflate=inflate,expand=nr_expand)})
                    port[rkey][hsdname][k].set_runkey(rkey).set_hsdname(hsdname)
                    port[rkey][hsdname][k].set_logicthresh(1<<12)
                    if is_fex:
                        port[rkey][hsdname][k].setRollOn((3*int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.xpre))>>2) # guessing that 3/4 of the pre and post extension for threshold crossing in fex is a good range for the roll on and off of the signal
                        port[rkey][hsdname][k].setRollOff((3*int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.xpost))>>2)
                    else:
                        port[rkey][hsdname][k].setRollOn(1<<6) 
                        port[rkey][hsdname][k].setRollOff(1<<6)
            else:
                runhsd = False

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
        wv = {}
        wv_logic = {}
        init = True 
        vsize = 0
        hsdEvents = []


        eventnum:int = 0 # later move this to outside the runs loop and let eventnum increase over all of the serial runs.

        for evt in run.events():
            completeEvent:List[bool] = [True]
            if eventnum > nshots:
                break
            #if eventnum%50==0: print('running event %i'%eventnum)
            ## test hsds
            if runhsd and bool(np.prod(completeEvent)):
                for i,hsdname in enumerate(hsds[rkey].keys()):
                    if (type(hsds[rkey][hsdname]) != None): 
                        for key in chankeys[rkey][hsdname]: # here key means 'port number'
                            if is_fex:
                                if (hsds[rkey][hsdname].raw.peaks(evt) != None):
                                    completeEvent += [port[rkey][hsdname][key].test(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0])]
                                else:
                                    print(eventnum,'hsds[%i][%s].raw.peaks(evt) is None'%(rkey,hsdname))
                                    completeEvent += [False]
                            else:
                                if (hsds[rkey][hsdname].raw.waveforms(evt) != None):
                                    completeEvent += [port[rkey][hsdname][key].test(hsds[rkey][hsdname].raw.waveforms(evt)[ key ][0])]
                                else:
                                    print(eventnum,'hsds[%i][%s].raw.waveforms(evt) is None'%(rkey,hsdname))
                                    completeEvent += [False]
                    else:
                        print(eventnum,'hsds is None')
                        completeEvent += [False]



            ## finish testing all detectors to measure ##
            ## before processing ##

            ## process hsds
            if runhsd and bool(np.prod(completeEvent)):
                for hsdname in hsds[rkey].keys():
                    ''' HSD-Abaco section '''
                    for key in chankeys[rkey][hsdname]: # here key means 'port number'
                        nwins:int = 1
                        xlist:List[int] = []
                        slist:List[ List[int] ] = []
                        baseline = np.uint32(0)
                        if is_fex:
                            nwins = len(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][0])
                            if nwins >2 : # always reports the start of and the end of the fex active window.
                                baseline = np.sum(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][1][0].astype(np.uint32))
                                baseline //= len(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][1][0])
                                for i in range(1,nwins-2):
                                    xlist += [ hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][0][i] ]
                                    slist += [ np.array(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][1][i],dtype=np.int32) ]
                        else:
                            slist += [ np.array(hsds[rkey][hsdname].raw.waveforms(evt)[ key ][0] , dtype=np.int16) ] # presumably 12 bits unsigned input, cast as int16_t since will immediately in-place subtract baseline
                            xlist += [0]
                        port[rkey][hsdname][key].set_baseline(baseline).process(slist,xlist) # this making a list out of the waveforms is to accommodate both the fex and the non-fex with the same Port object and .process() method.

            ## redundant events vec
            if bool(np.prod(completeEvent)):
                if runhsd:
                    hsdEvents += [eventnum]

            if init:
                init = False
                for hsdname in port[rkey].keys():
                    for key in port[rkey][hsdname].keys():
                        port[rkey][hsdname][key].set_initState(False)

            eventnum += 1


            if runhsd:
                if eventnum<2:
                    for hsdname in hsds[rkey].keys():
                        print('ports = %s'%([k for k in chankeys[rkey][hsdname].keys()]))
                if eventnum<100 and eventnum%10==0: 
                    for hsdname in hsds[rkey].keys():
                        print('working event %i,\tnedges = %s'%(eventnum,[port[rkey][hsdname][k].getnedges() for k in chankeys[rkey][hsdname]] ))
                elif eventnum<1000 and eventnum%25==0: 
                    for hsdname in hsds[rkey].keys():
                        print('working event %i,\tnedges = %s'%(eventnum,[port[rkey][hsdname][k].getnedges() for k in chankeys[rkey][hsdname]] ))
                else:
                    if eventnum%500==0: 
                        for hsdname in hsds[rkey].keys():
                            print('working event %i,\tnedges = %s'%(eventnum,[port[rkey][hsdname][k].getnedges() for k in chankeys[rkey][hsdname]] ))


            if eventnum > 1 and eventnum <1000 and eventnum%100==0:
                with h5py.File(outnames[rkey],'w') as f:
                    print('writing to %s'%outnames[rkey])
                    if runhsd:
                        Port.update_h5(f,port,hsdEvents)

            elif eventnum>900 and eventnum%1000==0:
                with h5py.File(outnames[rkey],'w') as f:
                    print('writing to %s'%outnames[rkey])
                    if runhsd:
                        Port.update_h5(f,port,hsdEvents)
        # end event loop

 

    print("Hello, I'm done now.  Have a most excellent day!")
    return





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
