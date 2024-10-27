#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.7-py39/bin/python3
import numpy as np
import sys
import h5py
import re
import matplotlib.pyplot as plt
from Quantizers import Quantizer
from utils import inlims

def main():
    if len(sys.argv)<6:
        print('syntax: quantizeToFVLScor.py <ntofbins> <nvlsbins> <gmdlow> <gmdhigh> <fnames>')
        return

    ## swithch to control uniform or nonuniform binning
    unonu = 'nonuniform'
    qknob = float(0.0)
    target = 'Ne'
    plotting = False

    donorm = False
    fnames = sys.argv[5:]
    ntofbins = np.uint32(sys.argv[1])
    nvlsbins = np.uint32(sys.argv[2])
    gmdlow,gmdhigh = np.uint16(sys.argv[3]),np.uint16(sys.argv[4])
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
    vlsquant_3rd = Quantizer(style='wave',nbins=nvlsbins)
    vlsquant_2nd = Quantizer(style='wave',nbins=nvlsbins)

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
                vlsdata_3rd = list(f['vls']['data'][()][:,:1<<9])
                vlsdata_2nd = list(f['vls']['data'][()][:,1<<10:])
            else:
                for k in portkeys:
                    offsetTofs = np.uint64(len(tofs[k]))
                    addresses[k] += [offsetTofs+np.uint64(a) for a in f[k]['addresses'][()]]
                    nedges[k] += list(f[k]['nedges'][()])
                    tofs[k] += list(f[k]['tofs'][()].astype(np.uint64))
                gmdens += list(f['gmd']['gmdenergy'][()])
                vlsdata_3rd += list(f['vls']['data'][()][:,:1<<9])
                vlsdata_2nd += list(f['vls']['data'][()][:,1<<10:])


    gmdquant.setbins(data=gmdens)
    gmdnorm = np.zeros(gmdquant.getnbins())


    usetofs = {}
    for k in portkeys:
        usetofs.update({k:[]})

    usevls_2nd = []
    usevls_3rd = []

## building valid tofs and vls from windowed gmd values
    for shot,gmden in enumerate(gmdens):
        if inlims(gmdens[shot],gmdlow,gmdhigh):

            if len(usevls_2nd)>0:
                usevls_2nd += [vlsdata_2nd[shot]]
            else:
                usevls_2nd = [vlsdata_2nd[shot]]

            if len(usevls_3rd)>0:
                usevls_3rd += [vlsdata_3rd[shot]]
            else:
                usevls_3rd = [vlsdata_3rd[shot]]

            for k in portkeys:
                a = addresses[k][shot]
                n = nedges[k][shot]
                if len(usetofs[k])>0:
                    usetofs[k] += list(tofs[k][a:a+n])
                else:
                    usetofs[k] = list(tofs[k][a:a+n])

    for k in portkeys:
        print(len(usetofs[k]))
        if len(usetofs[k])>1:
            quants[k].setbins(data=usetofs[k],knob=qknob)
    vlsquant_3rd.setbins(data=usevls_3rd)
    vlsquant_2nd.setbins(data=usevls_2nd)

    vlshist_2nd = []
    vlshist_3rd = []

## building quantized histograms
    for shot,gmden in enumerate(gmdens):
        if inlims(gmdens[shot],gmdlow,gmdhigh):
            vlshist_3rd += [vlsquant_3rd.histogram(vlsdata_3rd[shot])]
            vlshist_2nd += [vlsquant_2nd.histogram(vlsdata_2nd[shot])]
            for k in portkeys:
                a = addresses[k][shot]
                n = nedges[k][shot]
                hist[k] += list(quants[k].histogram(tofs[k][a:a+n]))


## setting the output filename
#/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_santafe/h5files/hits.tmox42619.run_088.h5
    outname = './test.h5'
    m = re.search('(^.*h5files)/hits\.(\w+)\..*\.h5',fnames[0])
    if m:
        if quants[portkeys[0]].getstyle() =='santafe':
            outname = '%s/quantHistVls_%s_runs'%(m.group(1),unonu) + '-'.join([r for r in runlist]) + '.%s.qknob%.3f.%s.h5'%(target,qknob,m.group(2))
        else:
            outname = '%s/quantHistVls_%s_runs'%(m.group(1),unonu) + '-'.join([r for r in runlist]) + '.%s.%s.h5'%(target,m.group(2))


## writing everything to an h5 file
    with h5py.File(outname,'w') as o:
        o.create_dataset('gmdwin',data = (gmdlow,gmdhigh))
        vgrp = o.create_group('vls')
        vgrp.create_dataset('hist_2nd',data = vlshist_2nd)
        vgrp.create_dataset('hist_3rd',data = vlshist_3rd)
        vgrp.create_dataset('qbins_3rd',data = vlsquant_3rd.binedges())
        vgrp.create_dataset('qbins_2nd',data = vlsquant_2nd.binedges())
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
            plt.savefig('./figs/quantizedSantaFe_qknob%.1f_runs%s_%s_%s_gmd_%i-%i.hist.png'%(qknob,'-'.join(runlist),unonu,k,gmdlow,gmdhigh))
            if k=='port_12':
                plt.show()

    return

if __name__=='__main__':
    main()

