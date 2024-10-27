#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3


import psana
import numpy as np
import sys
import re
import h5py
from scipy.fftpack import dct,dst

from Ports import *
from Ebeam import *
from Vls import *
from utils import *


def fillconfigs(cfgname):
    params = {'chans':{},'t0s':{},'logicthresh':{}}
    with h5py.File(cfgname,'r') as f:
        params['inflate'] = f.attrs['inflate']
        params['expand'] = f.attrs['expand']
        for p in f.keys():
            m = re.search('^\w+_(\d+)$',p)
            if m:
                k = int(m.group(1))
                params['chans'][k] = f[p].attrs['hsd']
                params['t0s'][k] = f[p].attrs['t0']
                params['logicthresh'][k] = f[p].attrs['logicthresh']
    return params

def main():
        ############################################
        ###### Change this to your output dir ######
        ############################################
    scratchdir = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_Christos/h5files'

    if len(sys.argv)<3:
        print('syntax: ./waves2h5.py <nshots> <expname> <list of run numbers>')
        return
    expname = sys.argv[2]
    nshots = int(sys.argv[1])
    runnums = [int(run) for run in sys.argv[3:]]
    outnames = ['%s/waves.%s.run_%03i.h5'%(scratchdir,expname,r) for r in runnums]

    _=[print('starting analysis exp %s for run %i'%(expname,int(r))) for r in runnums]
    cfgname = '%s/%s.configs.h5'%(scratchdir,expname)
    params = fillconfigs(cfgname)
    chans = params['chans']
    t0s = params['t0s']
    logicthresh = params['logicthresh']

    nr_expand = params['expand']


    port = [{} for r in runnums] 
    inflate = params['inflate']

    for r in range(len(runnums)):
        for key in chans.keys():
            port[r][key] = Port(key,chans[key],t0=t0s[key],logicthresh=logicthresh[key],inflate=inflate,expand=nr_expand,nrolloff=2**6)

    ds = [psana.DataSource(exp=expname,run=r) for r in runnums]

    runs = [next(ds[r].runs()) for r in range(len(runnums))]
    for r in range(len(runnums)):
        print(runs[r].detnames)
    runhsd=True
    hsds = []
    for r in range(len(runnums)):
        eventnum = 0
        print('processing run %i'%runnums[r])
        if runhsd and 'hsd' in runs[r].detnames:
            hsds += [runs[r].Detector('hsd')]

        wv = {}
        wv_logic = {}

        hsdEvents = []

        init = True 

        print('chans: ',chans)
        for evt in runs[r].events():
            if eventnum > nshots:
                break

            if runhsd:
    
                ''' HSD-Abaco section '''
                goodevents = 0
                for key in chans.keys(): # here key means 'port number'
                    #try:
                    s = np.array(hsds[r].raw.waveforms(evt)[ chans[key] ][0] , dtype=np.int16) 
                    if port[r][key].processChristos(s):
                        goodevents += 1
                    else:
                        print(eventnum, 'hsd process == False for %s'%key)
                        continue
                    '''
                    except:
                        print(eventnum, 'failed hsd for some reason')
                        continue
                    '''
                hsdEvents += [eventnum]

                if init:
                    init = False
                    for key in chans.keys():
                        port[r][key].set_initState(False)


                if eventnum<2:
                        print('ports = %s'%([k for k in chans.keys()]))
                if eventnum<100:
                    if eventnum%10==0: 
                        print('working event %i,\tnedges = %s'%(eventnum,[port[r][k].getnedges() for k in chans.keys()] ))
                elif eventnum<1000:
                    if eventnum%100==0: 
                        print('working event %i,\tnedges = %s'%(eventnum,[port[r][k].getnedges() for k in chans.keys()] ))
                else:
                    if eventnum%1000==0: 
                        print('working event %i,\tnedges = %s'%(eventnum,[port[r][k].getnedges() for k in chans.keys()] ))


            if eventnum > 1 and eventnum <1000 and eventnum%100==0:
                with h5py.File(outnames[r],'w') as f:
                    print('writing to %s'%outnames[r])
                    if runhsd:
                        Port.update_h5(f,port[r],hsdEvents,chans)

            elif eventnum>900 and eventnum%1000==0:
                with h5py.File(outnames[r],'w') as f:
                    print('writing to %s'%outnames[r])
                    if runhsd:
                        Port.update_h5(f,port[r],hsdEvents,chans)
            eventnum += 1
 

        with h5py.File(outnames[r],'w') as f:
            print('writing to %s'%outnames[r])
            if runhsd:
                Port.update_h5(f,port[r],hsdEvents,chans)

        print('Finished with run %i'%runnums[r])
    print("Hello, I'm done now!")
    return

if __name__ == '__main__':
    main()
