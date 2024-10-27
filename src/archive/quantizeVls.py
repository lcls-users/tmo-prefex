#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.7-py39/bin/python3
import numpy as np
import sys
import h5py
import re
import matplotlib.pyplot as plt
from Quantizers import Quantizer

def main():
    if len(sys.argv)<4:
        print('syntax: quantizeVls.py <ntofbins> <nvlsbins> <ngmdbins> <fnames>')
        return

    donorm = False
    fnames = sys.argv[4:]
    ntofbins = np.uint32(sys.argv[1])
    nvlsbins = np.uint16(sys.argv[2])
    vlsorder = 'third' # could also be 'second' for Neon. For run 95 with 850eV nominal energy (840eV actual) there is second only, third fell off the andor., NNO is only second order
    ngmdbins = np.uint16(sys.argv[3])
    gmdquant = Quantizer(style='nonuniform',nbins = ngmdbins)
    vlsquant = Quantizer(style='nonuniform',nbins = nvlsbins)
    tofs = {} 
    addresses = {} 
    nedges = {} 
    quants = {}
    hist = {}
    vlscenters = []
    vlssums = []
    gmdens = []
    portkeys = []
    for fname in fnames:
        vlspitchcorrect = 0
        m = re.search('run_(\d+)',fname)
        if m:
            vlspitchcorrect = 141 if int(m.group(1))<87 and int(m.group(1))>81 else 0
        with h5py.File(fname,'r') as f:
            portkeys = [k for k in f.keys() if (re.search('port',k) and not re.search('_16',k) and not re.search('_2',k))]
            if len(quants.keys())==0:
                for k in portkeys:
                    quants[k] = Quantizer(style='nonuniform',nbins=ntofbins)
                    tofs[k] = list(f[k]['tofs'][()])
                    addresses[k] = list(f[k]['addresses'][()].astype(np.uint64))
                    nedges[k] = list(f[k]['nedges'][()])
                    hist[k] = np.zeros((nvlsbins,ngmdbins,ntofbins),dtype=float)
                vlscenters = list(f['vls']['centroids'][()]+vlspitchcorrect)
                vlssums = list(f['vls']['sum'][()])
                gmdens = list(f['gmd']['gmdenergy'][()])
            else:
                for k in portkeys:
                    offsetTofs = np.uint64(len(tofs[k]))
                    addresses[k] += [offsetTofs+np.uint64(a) for a in f[k]['addresses'][()]]
                    nedges[k] += list(f[k]['nedges'][()])
                    tofs[k] += list(f[k]['tofs'][()])
                vlscenters += list(f['vls']['centroids'][()]+vlspitchcorrect)
                vlssums += list(f['vls']['sum'][()])
                gmdens += list(f['gmd']['gmdenergy'][()])

    if len(tofs[k])>1:
        _=[print('%s\t%i\t%i'%(k,len(tofs[k]),addresses[k][-1]+nedges[k][-1])) for k in portkeys]

    for k in portkeys:
        if len(tofs[k])>1:
            quants[k].setbins(data=tofs[k])

    vlsquant.setbins(data=vlscenters)
    plt.step(vlsquant.bincenters(),vlsquant.histogram(data=vlscenters)/vlsquant.binwidths())
    plt.title('vls')
    plt.xlabel('vls pixels')
    plt.ylabel('shots/pixel')
    plt.show()

    gmdquant.setbins(data=gmdens)
    plt.step(gmdquant.bincenters(),gmdquant.histogram(data=gmdens)/gmdquant.binwidths())
    plt.title('gmd')
    plt.xlabel('gmd value [uJ]')
    plt.ylabel('shots/uJ')
    plt.show()

    vlsnorm = np.zeros(vlsquant.getnbins())
    gmdnorm = np.zeros(gmdquant.getnbins())

    assert len(gmdens)==len(vlscenters)

    #/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_vernier_1000vlsthresh/h5files/hits.tmox42619.run_088.h5
    outname = './test.h5'
    m = re.search('(^.*h5files)/hits\.(\w+)\..*\.h5',fnames[0])
    if m:
        outname = '%s/quantHist.%s.h5'%(m.group(1),m.group(2))

    for shot in range(len(gmdens)):
        gmdnorm[gmdquant.getbin(gmdens[shot])] += gmdens[shot]
        vlsnorm[vlsquant.getbin(vlscenters[shot])] += vlssums[shot]
        for k in portkeys:
            a = addresses[k][shot]
            n = nedges[k][shot]
            hist[k][vlsquant.getbin(vlscenters[shot]),gmdquant.getbin(gmdens[shot]),:] += quants[k].histogram(tofs[k][a:a+n]).astype(float)


    with h5py.File(outname,'w') as o:
        gmdgrp = o.create_group('gmd')
        gmdgrp.create_dataset('norm',data = gmdnorm)
        gmdgrp.create_dataset('qbins',data = gmdquant.binedges())
        vlsgrp = o.create_group('vls')
        vlsgrp.create_dataset('norm',data = vlsnorm)
        vlsgrp.create_dataset('qbins',data = vlsquant.binedges())
        for k in portkeys:
            kgrp = o.create_group(k)
            kgrp.create_dataset('hist',data= hist[k])
            kgrp.create_dataset('qbins',data= quants[k].binedges())
    return

if __name__=='__main__':
    main()

    '''
    crop=2
    ycrop=1
    fig1,ax = plt.subplots(2,4,figsize=(18,9))
    for i,k in enumerate(portkeys):
        if len(tofs[k])>1:
            X,Y = np.meshgrid(quants[k].binedges()[1:-crop-1],np.arange(nvlsbins+1-ycrop))
            if donorm:
                for v in range(nvlsbins):
                    if vlsnorm[v]>0:
                        hist[k][v,:] /= vlsnorm[v]
                    else:
                        hist[k][v,:] *= 0
            #ax[i//4,i%4].pcolor(X,Y,hist[k][:-ycrop,1:-crop])#,origin='lower')
            ax[i//4,i%4].pcolor(X,Y,np.log2(hist[k][:-ycrop,1:-crop]))#,origin='lower')
            ax[i//4,i%4].set_title('%s'%k)
            ax[i//4,i%4].set_xlabel('tofs')
            ax[i//4,i%4].set_ylabel('vls')
    plt.savefig('Figure_1_tofs_oneto%i.png'%crop)
    plt.show()

    fig2,ax = plt.subplots(2,4,figsize=(18,9))
    for i,k in enumerate(portkeys):
        if len(tofs[k])>1:
            X,Y = np.meshgrid(np.arange(len(quants[k].binedges())-crop-1),np.arange(nvlsbins+1-ycrop))
            if donorm:
                for v in range(nvlsbins):
                    if vlsnorm[v]>0:
                        hist[k][v,:] /= vlsnorm[v]
                    else:
                        hist[k][v,:] *= 0
            ax[i//4,i%4].pcolor(X,Y,np.log2(hist[k][:-ycrop,1:-crop]))#,origin='lower')
            ax[i//4,i%4].set_title('%s'%k)
            ax[i//4,i%4].set_xlabel('qbins')
            ax[i//4,i%4].set_ylabel('vls')
    plt.savefig('Figure_1_qbins_oneto%i.png'%crop)
    plt.show()
        #outname = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_vernier/ascii/test_%s_hist.dat'%(k)

    '''


    '''
    for k in ['port_0','port_1','port_14','port_5','port_12']:
        fg,ax = plt.subplots(1,1,figsize=(5,4))
        X,Y = np.meshgrid(quants[k].binedges()[1:-crop],np.arange(nvlsbins+1-ycrop))
        ax.pcolor(X,Y,hist[k][:-ycrop,1:-crop])
        ax.set_title('%s'%k)
        ax.set_ylabel('vls [arb]')
        ax.set_xlabel('ToF [arb]')
        plt.savefig('%s_tofs_oneto%i.png'%(k,crop))
        #plt.savefig('%s_tofs_oneto%i_raw.png'%(k,crop))
        plt.show()

    for k in ['port_0','port_1','port_14','port_5','port_12']:
        fg,ax = plt.subplots(1,1,figsize=(5,4))
        X,Y = np.meshgrid(np.arange(len(quants[k].binedges())-1-crop),np.arange(nvlsbins+1-ycrop))
        ax.pcolor(X,Y,hist[k][:-ycrop,1:-crop])
        ax.set_title('%s'%k)
        ax.set_ylabel('vls [arb]')
        ax.set_xlabel('Qbins [index]')
        plt.savefig('%s_qbins_oneto%i.png'%(k,crop))
        #plt.savefig('%s_qbins_oneto%i_raw.png'%(k,crop))
        plt.show()
        '''



