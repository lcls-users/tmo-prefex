#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.7-py39/bin/python3
import numpy as np
import sys
import h5py
import re
import matplotlib.pyplot as plt
from Quantizers import Quantizer

def main():
    if len(sys.argv)<3:
        print('syntax: quantizeGmd.py <ntofbins> <ngmdbins> <fnames>')
        return

    donorm = True
    fnames = sys.argv[3:]
    ntofbins = np.uint32(sys.argv[1])
    ngmdbins = np.uint32(sys.argv[2])
    gmdquant = Quantizer(style='nonuniform',nbins = ngmdbins)
    tofs = {} 
    addresses = {} 
    nedges = {} 
    quants = {}
    hist = {}
    gmdens = []
    portkeys = []
    for fname in fnames:
        with h5py.File(fname,'r') as f:
            portkeys = [k for k in f.keys() if (re.search('port',k))]# and not re.search('_16',k) and not re.search('_2',k))] # keeping the bare MCP ports 2 and 16 here
            #portkeys = [k for k in f.keys() if (re.search('port',k) and not re.search('_16',k) and not re.search('_2',k))] # keeping the bare MCP ports 2 and 16 here
            if len(quants.keys())==0:
                for k in portkeys:
                    quants[k] = Quantizer(style='nonuniform',nbins=ntofbins)
                    tofs[k] = list(f[k]['tofs'][()])
                    addresses[k] = list(f[k]['addresses'][()].astype(np.uint64))
                    nedges[k] = list(f[k]['nedges'][()])
                    hist[k] = np.zeros((ngmdbins,ntofbins),dtype=float)
                gmdens = list(f['gmd']['gmdenergy'][()])
            else:
                for k in portkeys:
                    offsetTofs = np.uint64(len(tofs[k]))
                    addresses[k] += [offsetTofs+np.uint64(a) for a in f[k]['addresses'][()]]
                    nedges[k] += list(f[k]['nedges'][()])
                    tofs[k] += list(f[k]['tofs'][()])
                gmdens += list(f['gmd']['gmdenergy'][()])
        _= [print(k,len(gmdens),len(addresses[k])) for k in portkeys]

    if len(tofs[k])>1:
        _=[print('%s\t%i\t%i'%(k,len(tofs[k]),addresses[k][-1]+nedges[k][-1])) for k in portkeys]

    gmdquant.setbins(data=gmdens)
    for k in portkeys:
        if len(tofs[k])>1:
            quants[k].setbins(data=tofs[k])

    #h,b = np.histogram(gmdens,100)
    h,b = np.histogram(gmdens,gmdquant.binedges())
    plt.plot(b[:-1],1/gmdquant.binwidths(),'.')
    plt.show()


    gmdnorm = np.zeros(gmdquant.getnbins())
    for shot,gmden in enumerate(gmdens):
        gmdnorm[gmdquant.getbin(gmden)] += gmden
        for k in portkeys:
            a = addresses[k][shot]
            n = nedges[k][shot]
            hist[k][gmdquant.getbin(gmden),:] += quants[k].histogram(tofs[k][a:a+n]).astype(float)

    nrows = 2
    ncols = len(portkeys)//nrows
    fig1,ax = plt.subplots(nrows,ncols,figsize=(18,9))

    with h5py.File('%s.counts.qtofs.qgmd.h5'%fnames[0],'w') as o:
        ggrp = o.create_group('gmd')
        ggrp.create_dataset('bins',data=gmdquant.binedges())
        ggrp.create_dataset('norm',data=gmdnorm)
        for i,k in enumerate(portkeys):
            kgrp = o.create_group(k)
            if len(tofs[k])>1:
                kgrp.create_dataset('quantbins',data=quants[k].binedges())
                #X,Y = np.meshgrid(quants[k].binedges(),gmdquant.binedges())
                X,Y = np.meshgrid(np.arange(len(quants[k].binedges())-1),gmdquant.binedges()[:-1])
                if donorm:
                    res = np.zeros(hist[k].shape,dtype=float)
                    for b in range(gmdquant.getnbins()):
                        if gmdnorm[b]>0:
                            res[b,:] = hist[k][b,:].astype(float) / float(gmdnorm[b])
                            res[b,:] /= quants[k].binwidths()
                            #res[b,:] /= gmdquant.binwidths()[b]
                    ax[i//ncols,i%(ncols)].pcolor(X[1:,:],Y[1:,:],res[1:,:])#,origin='lower')
                    #ax[i//ncols,i%(ncols)].pcolor(X[1:,:],Y[1:,:],np.log2(res[1:,:]))#,origin='lower')
                    #ax[i//ncols,i%ncols].pcolor(X[1:,len(quants[k].binedges())//2:],Y[1:,len(quants[k].binedges())//2:],res[1:,len(quants[k].binedges())//2:])#,origin='lower')
                    #ax[i//ncols,i%ncols].pcolor(X[1:,:len(quants[k].binedges())//2],Y[1:,:len(quants[k].binedges())//2],res[1:,:len(quants[k].binedges())//2])#,origin='lower')
                    ax[i//ncols,i%ncols].set_title('%s'%k)
                    ax[i//ncols,i%ncols].set_xlabel('quantized tofs')
                    ax[i//ncols,i%ncols].set_ylabel('gmd [uJ]')
                else:
                    ax[i//ncols,i%ncols].pcolor(X,Y,hist[k][1:,:])#,origin='lower')
                    #ax[i//ncols,i%ncols].pcolor(X,Y,np.log2(hist[k]))#,origin='lower')
                    ax[i//ncols,i%ncols].set_title('%s'%k)
                    ax[i//ncols,i%ncols].set_xlabel('quantized tofs')
                    ax[i//ncols,i%ncols].set_ylabel('gmd [uJ]')
                kgrp.create_dataset('hist',data=hist[k])
        plt.savefig('Figure_all_qtofs_gmd.png')
        plt.show()
    return



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

    return

if __name__=='__main__':
    main()

