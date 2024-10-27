#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.7-py39/bin/python3
import numpy as np
import sys
import h5py
import re
import matplotlib.pyplot as plt
from Quantizers import Quantizer
from utils import inlims

def main():
    if len(sys.argv)<5:
        print('syntax: quantizeSantaFe.py <ntofbins> <gmdlow> <gmdhigh> <fnames>')
        return

    ## swithch to control uniform or nonuniform binning
    unonu = 'nonuniform'
    qknob = float(0.0)
    target = 'unk'
    target = 'Ne'
    target = 'NNO'
    plotting = False

    donorm = False
    fnames = sys.argv[4:]
    ntofbins = np.uint32(sys.argv[1])
    vlsorder = 'second' # second for the NNO data
    gmdlow,gmdhigh = np.uint16(sys.argv[2]),np.uint16(sys.argv[3])
    tofs = {} 
    addresses = {} 
    nedges = {} 
    quants = {}
    hist = {}
    rate = {}
    gmdquant = Quantizer(style='nonuniform',nbins=1<<6)
    gmdens = []
    portkeys = []
    runlist = []
    for fname in fnames:
        m = re.search('run_(\d+)',fname)
        if m:
            runlist += [m.group(1)]
        with h5py.File(fname,'r') as f:
            portkeys = [k for k in f.keys() if (re.search('port',k) and not re.search('_16',k) and not re.search('_2',k))]
            if len(quants.keys())==0:
                for k in portkeys:
                    quants[k] = Quantizer(style=unonu,nbins=ntofbins)
                    tofs[k] = list(f[k]['tofs'][()].astype(np.uint64))
                    addresses[k] = list(f[k]['addresses'][()].astype(np.uint64))
                    nedges[k] = list(f[k]['nedges'][()])
                    hist[k] = []
                gmdens = list(f['gmd']['gmdenergy'][()])
            else:
                for k in portkeys:
                    offsetTofs = np.uint64(len(tofs[k]))
                    addresses[k] += [offsetTofs+np.uint64(a) for a in f[k]['addresses'][()]]
                    nedges[k] += list(f[k]['nedges'][()])
                    tofs[k] += list(f[k]['tofs'][()].astype(np.uint64))
                gmdens += list(f['gmd']['gmdenergy'][()])


    gmdquant.setbins(data=gmdens)
    gmdnorm = np.zeros(gmdquant.getnbins())

    #/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_santafe/h5files/hits.tmox42619.run_088.h5
    outname = './test.h5'
    m = re.search('(^.*h5files)/hits\.(\w+)\..*\.h5',fnames[0])
    if m:
        outname = '%s/quantHist_%s_runs'%(m.group(1),unonu) + '-'.join([r for r in runlist]) + '.%s.qknob%.3f.%s.h5'%(target,qknob,m.group(2))

    usetofs = {}
    for k in portkeys:
        usetofs.update({k:[]})

## building valid tofs from windowed gmd values
    for shot,gmden in enumerate(gmdens):
        if inlims(gmdens[shot],gmdlow,gmdhigh):
            for k in portkeys:
                a = addresses[k][shot]
                n = nedges[k][shot]
                if len(usetofs[k])>1:
                    usetofs[k] += list(tofs[k][a:a+n])
                else:
                    usetofs[k] = list(tofs[k][a:a+n])

    for k in portkeys:
        print(len(usetofs[k]))
        if len(usetofs[k])>1:
            quants[k].setbins(data=usetofs[k],knob=qknob)


## building quantized histogram
    for shot,gmden in enumerate(gmdens):
        if inlims(gmdens[shot],gmdlow,gmdhigh):
            for k in portkeys:
                a = addresses[k][shot]
                n = nedges[k][shot]
                hist[k] += list(quants[k].histogram(tofs[k][a:a+n]))

    with h5py.File(outname,'w') as o:
        o.create_dataset('gmdwin',data = (gmdlow,gmdhigh))
        for k in portkeys:
            kgrp = o.create_group(k)
            kgrp.create_dataset('hist',data= np.array(hist[k]).reshape(len(hist[k])//quants[k].getnbins(),-1).T)
            kgrp.create_dataset('qbins',data= quants[k].binedges())
            print(k,np.max(hist[k]))

    if plotting:
        for k in hist.keys():
            fig,ax = plt.subplots(1,1,figsize=(8,8))
            im = ax.pcolor(np.array(hist[k]).reshape(len(hist[k])//quants[k].getnbins(),-1).T,vmax=2)
            if not unonu=='nonuniform':
                ax.set_ylim(0,1<<7)
            ax.set_xlabel('shot number (pulse energy selected)')
            ax.set_ylabel('quantized ToF bin')
            ax.set_title('%s binning: %s: %i-%iuJ'%(unonu,k,gmdlow,gmdhigh))
            plt.colorbar(im,ax=ax)
            plt.savefig('./figs/quantized_qknob%.1f_runs%s_%s_%s_gmd_%i-%i.hist.png'%(qknob,'-'.join(runlist),unonu,k,gmdlow,gmdhigh))
            if k=='port_12':
                plt.show()

    return

if __name__=='__main__':
    main()

