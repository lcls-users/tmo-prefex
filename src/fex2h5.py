#!/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps-4.6.3/bin/python3

import psana
from collections import deque
import numpy as np
import sys
import re
import h5py
import os
import socket
#import objsize
from typing import Type,List

from Ports import *
from Ebeam import *
#from Vls import *
from Gmd import *
from Spect import *
from Atm import *
from Config import Config
from utils import *
import yaml

def main(nshots:int,runnums:List[int]):
  
    outnames = {}
    expname = os.environ.get('expname')

    _=[print('starting analysis exp %s for run %i'%(expname,int(r))) for r in runnums]

    #######################
    #### CONFIGURATION ####
    #######################
    cfgname:str = '%s/config.yaml'%(os.environ.get('configpath'))
    is_fex:bool = True 
    inflate:int = 2
    expand:int = 2

    goose:int = 281 # on but mistimed
    laser:int = 280 # on and timed
    anylaser:int = 282 # on, either mistimed or timed

    runhsd=True
    rungmd=True
    runpiranha=True
    runatm=True
    runtiming=True

    runvls=False
    runebeam=False
    runxtcav=False


    ###################################
    #### Setting up the datasource ####
    ###################################
    
    port = {}
    chankeys = {}
    hsds = {}
    gmds = {}
    xray = {}
    piranhas = {}
    spect = {}
    atm = {}


    ds = psana.DataSource(exp=expname,run=runnums)
    detslist = {}
    hsdnames = {}
    gmdnames = {}
    pirnames = {}

    for r in runnums:
        chunk = 0
        run = next(ds.runs())
        rkey = run.runnum
        port.update({rkey:{}})
        hsds.update({rkey:{}})

        gmds.update({rkey:{}})
        xray.update({rkey:{}})

        piranhas.update({rkey:{}})
        spect.update({rkey:{}})
        atm.update({rkey:{}})

        

        chankeys.update({rkey:{}})
        detslist.update({rkey:[s for s in run.detnames]})
        outnames.update({rkey:'%s/hits.%s.%s.h5'%(os.environ.get('scratchpath'),os.environ.get('expname'),os.environ.get('runstr'))})
        hsdnames.update({rkey: [s for s in detslist[rkey] if re.search('hsd$',s)] })
        gmdnames.update({rkey: [s for s in detslist[rkey] if re.search('gmd$',s)] })
        pirnames.update({rkey: [s for s in detslist[rkey] if re.search('piranha$',s)] })

        for hsdname in hsdnames[rkey]:
            port[rkey].update({hsdname:{}})
            chankeys[rkey].update({hsdname:{}})
            if runhsd and hsdname in detslist[rkey]:
                hsds[rkey].update({hsdname:run.Detector(hsdname)})
                port[rkey].update({hsdname:{}})
                chankeys[rkey].update({hsdname:{}})
                for i,k in enumerate(list(hsds[rkey][hsdname].raw._seg_configs().keys())):
                    chankeys[rkey][hsdname].update({k:k}) # this we may want to replace with the PCIe address id or the HSD serial number.
                    port[rkey][hsdname].update({k:Port(k,chankeys[rkey][hsdname][k],inflate=inflate,expand=expand)})
                    port[rkey][hsdname][k].set_runkey(rkey).set_name(hsdname)
                    port[rkey][hsdname][k].set_logicthresh(-1*(1<<10))
                    if is_fex:
                        port[rkey][hsdname][k].setRollOn((3*int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.xpre))>>2) # guessing that 3/4 of the pre and post extension for threshold crossing in fex is a good range for the roll on and off of the signal
                        port[rkey][hsdname][k].setRollOff((3*int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.xpost))>>2)
                        port[rkey][hsdname][k].set_baseline(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.corr.baseline)
                        port[rkey][hsdname][k].set_logicthresh(int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.corr.baseline)-int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.ymax))
                    else:
                        port[rkey][hsdname][k].setRollOn(1<<6) 
                        port[rkey][hsdname][k].setRollOff(1<<6)
            else:
                runhsd = False

        for gmdname in gmdnames[rkey]:
            if rungmd and gmdname in detslist[rkey]:
                gmds[rkey].update({gmdname:run.Detector(gmdname)}) 
                xray[rkey].update({gmdname:Gmd()})
                xray[rkey][gmdname].set_runkey(rkey).set_name(gmdname)
                if re.search('xgmd',gmdname):
                    print('running xgmd damnit!')
                    xray[rkey][gmdname].set_unit('0.1uJ',scale=1e4)
            else:
                rungmd = False

        for pirname in pirnames[rkey]:
            if runpiranha and pirname in detslist[rkey]:
                piranhas[rkey].update({pirname:run.Detector(pirname)})
                if re.search('fzp',pirname):
                    print('running fzp too')
                    spect[rkey].update({pirname:Spect(thresh=(1<<3))})
                    spect[rkey][pirname].set_runkey(rkey).set_name(pirname)
                    spect[rkey][pirname].setProcessAlgo('piranha_centroid')
                if re.search('atm',pirname):
                    print('running atm also!!')
                    atm[rkey].update({pirname:Atm(thresh=(1<<3))})
                    atm[rkey][pirname].set_runkey(rkey).set_name(pirname)
                    atm[rkey][pirname].setProcessAlgo('piranha_edge')
            else:
                runpiranha = False

        init = True 
        hsdEvents = []
        gmdEvents = []
        spectEvents = []
        atmEvents = []
        eventnum:int = 0 # later move this to outside the runs loop and let eventnum increase over all of the serial runs.
        evrcodes:List(bool) = [False]*288


        '''
        # from Mona for timestamps
        timestamps = np.array([4194783241933859761,4194783249723600225,4194783254218190609,4194783258712780993], dtype=np.uint64)
        ds = DataSource(exp='tmoc00118', run=222, dir='/sdf/data/lcls/ds/prj/public01/xtc',timestamps=timestamps)
        myrun = next(ds.runs())
        opal = myrun.Detector('tmo_atmopal')
        print(nevt, evt.timestamp, img.shape)
        ts:np.uint64 = run.timestamp
        '''

        for eventnum,evt in enumerate(run.events()):
            completeEvent:List[bool] = [True]
            if eventnum > nshots:
                print("done")
                break

            #test readbacks for each of detectors for given event

            ## if failed test of piranha, can't do spectrum correlation
            if runpiranha and all(completeEvent):
                if piranhas[rkey] is not None:
                    for pirname in pirnames[rkey]:
                        if piranhas[rkey][pirname] is not None:
                            if re.search('fzp',pirname):
                                completeEvent += [spect[rkey][pirname].test(piranhas[rkey][pirname].raw.raw(evt)) ]
                            if re.search('atm',pirname):
                                completeEvent += [atm[rkey][pirname].test(piranhas[rkey][pirname].raw.raw(evt)) ]
                        else:
                            completeEvent += [False]

            ## if failed test of gmd, then can't normalize, so skip event.
            if rungmd and all(completeEvent):
                if gmds[rkey] is not None:
                    for gmdname in gmdnames[rkey]:
                        if gmds[rkey][gmdname] is not None:
                            completeEvent += [xray[rkey][gmdname].test(gmds[rkey][gmdname].raw.milliJoulesPerPulse(evt))]
                        else:
                            completeEvent += [False]
                else:
                    completeEvent += [False]

            ## if failed test of hsds, then NoneType for some detector, so skip event.
            if runhsd and all(completeEvent):
                for i,hsdname in enumerate(hsds[rkey].keys()):
                    if (hsds[rkey][hsdname] is not None): 
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


            #############################################
            ## finish testing all detectors to measure ##
            ############ before processing ##############
            #############################################

            ## process piranhas
            if runpiranha and all(completeEvent):
                for pirname in pirnames[rkey]:
                    if False and re.search('atm',pirname):
                        if evrcodes[goose] or not evrcodes[anylaser]:
                            atm[rkey][pirname].updateref(piranhas[rkey][pirname].raw.raw(evt))
                        else:
                            atm[rkey][pirname].process(piranhas[rkey][pirname].raw.raw(evt))
                    if re.search('fzp',pirname):
                        spect[rkey][pirname].process(piranhas[rkey][pirname].raw.raw(evt))

            ## process gmds
            if rungmd and all(completeEvent):
                for gmdname in gmdnames[rkey]:
                    xray[rkey][gmdname].process(gmds[rkey][gmdname].raw.milliJoulesPerPulse(evt))

            ## process hsds
            if runhsd and all(completeEvent):
                for hsdname in hsds[rkey].keys():
                    ''' HSD-Abaco section '''
                    for key in chankeys[rkey][hsdname]: # here key means 'port number'
                        nwins:int = 1
                        xlist:List[int] = []
                        slist:List[ List[int] ] = []
                        if is_fex:
                            nwins = len(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][0])
                            '''
                            if nwins < 3 : # always reports the start of and the end of the fex active window, and ignore if not a peak in there also
                                continue
                            '''
                            for i in range(nwins):
                                xlist += [ hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][0][i] ]
                                slist += [ np.array(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][1][i],dtype=np.int32) ]
                        elif eventnum > 100 and hsds[rkey][hsdname].raw.waveforms(evt) is not None:
                            slist += [ np.array(hsds[rkey][hsdname].raw.waveforms(evt)[ key ][0] , dtype=np.int16) ] 
                            xlist += [0]
                            wv = hsds[hkey][hsdname].raw.waveforms(evt)[ key ][0]
                            wvx = np.arange(wv.shape[0])
                            y = [hsd.raw.peaks(evt)[1][0][1][i] for i in range(len(hsd.raw.peaks(evt)[1][0][1]))]
                            x = [np.arange(hsd.raw.peaks(evt)[1][0][0][i],hsd.raw.peaks(evt)[1][0][0][i]+len(hsd.raw.peaks(evt)[1][0][1][i])) for i in range(len(y))]
                            plt.plot(wv)
                            _=[plt.plot(x[i],y[i]) for i in range(len(y))]
                            plt.show()
                        port[rkey][hsdname][key].process(slist,xlist) # this making a list out of the waveforms is to accommodate both the fex and the non-fex with the same Port object and .process() method.

            ## redundant events vec
            if all(completeEvent):
                if runhsd:
                    hsdEvents += [eventnum]
                if rungmd:
                    gmdEvents += [eventnum]
                if runpiranha:
                    spectEvents += [eventnum]
                    atmEvents += [eventnum]

            if init:
                init = False
                for hsdname in port[rkey].keys():
                    for key in port[rkey][hsdname].keys():
                        port[rkey][hsdname][key].set_initState(False)
                for gmdname in xray[rkey].keys():
                    xray[rkey][gmdname].set_initState(False)
                for pirname in spect[rkey].keys():
                    if re.search('fzp',pirname):
                        spect[rkey][pirname].set_initState(False)
                    if re.search('atm',pirname):
                        atm[rkey][pirname].set_initState(False)


            if runhsd:
                if eventnum<2:
                    for hsdname in hsds[rkey].keys():
                        print('ports = %s'%([k for k in chankeys[rkey][hsdname].keys()]))
                if eventnum<100 and eventnum%10==0: 
                    for hsdname in hsds[rkey].keys():
                        print('working event %i,\tnedges = %s'%(eventnum,[port[rkey][hsdname][k].getnedges() for k in chankeys[rkey][hsdname]] ))

                elif eventnum<1000 and eventnum%100==0: 
                    for hsdname in hsds[rkey].keys():
                        print('working event %i,\tnedges = %s'%(eventnum,[port[rkey][hsdname][k].getnedges() for k in chankeys[rkey][hsdname]] ))
                else:
                    if eventnum%1000==0: 
                        for hsdname in hsds[rkey].keys():
                            print('working event %i,\tnedges = %s'%(eventnum,[port[rkey][hsdname][k].getnedges() for k in chankeys[rkey][hsdname]] ))


            '''
            if eventnum>1 and eventnum%1000==0:
                filename_save = outnames[rkey][:-3]+".%04i.h5"%(chunk)
                with h5py.File(filename_save,'w') as f:
                    print('writing to %s'%filename_save)
                    if runhsd:
                        Port.update_h5(f,port,hsdEvents)
                    if rungmd:
                        Gmd.update_h5(f,xray,gmdEvents)
                    if runpiranha:
                        Spect.update_h5(f,spect,spectEvents)
                        '''
            
            if eventnum >= 10000 and eventnum % 10000==0:
                filename_save = outnames[rkey][:-3]+".%04i.h5"%(chunk)
                with h5py.File(filename_save,'w') as f:
                    print('writing to %s'%filename_save)
                    if runhsd:
                        Port.update_h5(f,port,hsdEvents)
                        hsdEvents.clear()
                        for name in port[rkey].keys():
                            for p in port[rkey][name].keys():
                                port[rkey][hsdname][p].reset()

                    if rungmd:
                        Gmd.update_h5(f,xray,gmdEvents)
                        gmdEvents.clear()
                        for name in xray[rkey].keys():
                            xray[rkey][name].reset()
                    
                    if runpiranha:
                        Spect.update_h5(f,spect,spectEvents)
                        Spect.update_h5(f,atm,atmEvents)
                        spectEvents.clear()
                        atmEvents.clear()
                        for name in spect[rkey].keys():
                            if re.search('fzp',pirname):
                                spect[rkey][pirname].reset()
                        for name in atm[rkey].keys():
                            if re.search('atm',pirname):
                                atm[rkey][pirname].reset()
                chunk += 1

        # end event loop

 

    print("Hello, I'm done now.  Have a most excellent day!")
    return






if __name__ == '__main__':
    if len(sys.argv)>3:
        nshots = int(sys.argv[1])
        runnums = [int(r) for r in list(sys.argv[3:])]
        scratchpath = os.environ.get('scratchpath')
        if not os.path.exists('scratchpath'):
            os.makedirs('scratchpath')
        main(nshots,runnums)
    else:
        print('Please be sure that expname is set in env and exported\n also give me a number of shots (-1 for all) and a list of run numbers.\n the output directory defaults to expt scratch directory and controlled by envvar as well.)')
