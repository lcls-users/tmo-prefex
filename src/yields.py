#!/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps-4.6.3/bin/python3

import psana
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Type,List

def main(expname,maxevents:np.uint32,runnums):
    thresh = 19000
    yields = {}
    gmds = {}
    xgmds = {}
    
    for runnum in runnums:
        ds = psana.DataSource(exp=expname,run=int(runnum))
        run = next(ds.runs())
        hsd = run.Detector('mrco_hsd')
        gmd = run.Detector('gmd')
        xgmd = run.Detector('xgmd')
        sumxgmd = float(0)
        sumgmd = float(0)
        chans = np.sort([k for k in hsd.raw._seg_chans().keys()])
        np.sort(chans)
        eventnum:np.uint64 = np.uint64(0)
        for ch in chans: 
            yields.update({ch:np.uint64(0)})
            gmds.update({ch:float(0)})
            xgmds.update({ch:float(0)})

        for evt in run.events():
            if eventnum%100 == 0:
                print('working eventnum %i'%(eventnum))
            evt = next(run.events())
            if eventnum > maxevents:
                print('breaking')
                break
            if (hsd.raw.peaks(evt) is None) or (xgmd.raw.milliJoulesPerPulse(evt) is None):
                print('skipping %i'%eventnum)
                continue
            sumxgmd += xgmd.raw.milliJoulesPerPulse(evt)
            for ch in chans:
                #if (hsd.raw.peaks(evt)[ch] is not None) and (hsd.raw.peaks(evt)[ch][0] is not None) and (hsd.raw.peaks(evt)[ch][0][1] is not None):
                if len(hsd.raw.peaks(evt)[ch][0][1])>2:
                    for i in range(1,len(hsd.raw.peaks(evt)[ch][0][1])-1):
                        if np.max(hsd.raw.peaks(evt)[ch][0][1][i])>thresh:
                            yields[ch] += 1
            eventnum += 1
        _= [print('chan: %i\t%i'%(ch,yields[ch])) for ch in chans]
        xvals = yields.keys()
        yvals = [float(yields[k])/sumxgmd for k in yields.keys()]
        plt.plot(xvals,yvals,label = 'run %i'%(run.runnum))
    plt.legend()
    plt.show()


    return

if __name__ == '__main__':
    if len(sys.argv)>3:
        print(sys.argv[0])
        print(sys.argv[1])
        print(sys.argv[2])
        print(sys.argv[3])
        main(sys.argv[2],int(sys.argv[1]),sys.argv[3:])
    else:
        print('please give an maxevents expname runnum list')
