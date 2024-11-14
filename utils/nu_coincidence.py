#!/usr/bin/python3
import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import statistics as stats

def main():
    path = sys.argv[1]
    filenames = np.sort([nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))])
    corrports = [['port_090','port_270'],['port_000','port_180']]
    #filenames = [nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))]

    hist = [np.zeros((1<<9,1<<9),dtype=np.uint64)]*len(corrports)
    h = {}
    falsecoince = {}
    sums = {}
    tofs = {}
    addresses = {}
    nedges = {}
    maxtof:np.uint32 = 1<<15
    t0=12000
    scale = 1

    fig,axs=plt.subplots(1,2,figsize=(24,12))

    for k in range(len(corrports)):
        for nameind,name in enumerate(filenames):
            with h5py.File(os.path.join(path,name),'r') as f:
                runstr = [k for k in f.keys()][0]
                detstr = 'mrco_hsd'
                hsd = f[runstr][detstr]
                fzp = f[runstr]['tmo_fzppiranha']
                xgmd = f[runstr]['xgmd']
                ports = [p for p in hsd.keys()]
                ens = xgmd['energy'][()]
                cents = fzp['centroids'][()]
        
    
                if corrports[k][0] not in h.keys():
                    h.update({corrports[k][0]:[int(0)]*(hist[k].shape[0])})
                    sums.update({corrports[k][0]:int(0)})
                if corrports[k][1] not in h.keys():
                    h.update({corrports[k][1]:[int(0)]*(hist[k].shape[1])})
                    sums.update({corrports[k][1]:int(0)})

                for p in corrports[k]:
                    tofs.update({p:hsd[p]['tofs'][()]})
                    addresses.update({p:hsd[p]['addresses'][()]})
                    nedges.update({p:hsd[p]['nedges'][()]})
                    for t in tofs[p]:
                        h[p][ max(0,min(len(h[p])-1, int(t-t0)>>1))] +=1
                    sums[p] = sum(h[corrports[k][0]])
                falsecoince.update({ k : np.outer(h[corrports[k][0]],h[corrports[k][1]]) })

                for i in range(len(addresses[corrports[k][0]])):
                    n1 = nedges[corrports[k][0]][i]
                    n2 = nedges[corrports[k][1]][i]
                    a1 = addresses[corrports[k][0]][i]
                    a2 = addresses[corrports[k][1]][i]
                    for t1 in tofs[corrports[k][0]][a1:a1+n1]:
                        for t2 in tofs[corrports[k][1]][a2:a2+n2]:
                            ind1 = max(0,min(hist[k].shape[0]-1, int(t1-t0)>>1))
                            ind2 = max(0,min(hist[k].shape[1]-1, int(t2-t0)>>1))
                            hist[k][ind1,ind2] += 1
        hscale = np.sum(falsecoince[k])
        fscale = np.sum(hist[k])
        print(hscale//fscale)

        if k == 0:
            im = axs[k].imshow(2*hist[k]-(falsecoince[k]*fscale//hscale),clim=(-1,10))
        else:
            im = axs[k].imshow(2*hist[k]-(falsecoince[k]*fscale//hscale),clim=(-10,100))
        axs[k].set_title('%s-%s'%(corrports[k][0],corrports[k][1]))
        fig.colorbar(im, orientation='vertical')
    plt.show()

    '''
    fig,axs=plt.subplots(1,2,figsize=(24,12))
    for k in range(len(corrports)):
        print(sum(h[corrports[k][0]]))
        #falsecoince[k] //= sum(h[corrports[k][0]])
        #falsecoince[k] //= sum(h[corrports[k][1]])
        im = axs[k].imshow(falsecoince[k],clim=(0,stats.mean(falsecoince[k])))
        axs[k].set_title('%s-%s'%(corrports[k][0],corrports[k][1]))
        fig.colorbar(im, orientation='vertical')
    plt.show()
    '''

    '''
    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].imshow(hist[p]) 
        axs[row,col].set_title(str(np.sum(hist[p]))) 
    plt.show()
    '''


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')


