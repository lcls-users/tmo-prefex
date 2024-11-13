#!/usr/bin/python3
import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import statistics as stats
from typing import List

def main():
    inpath = sys.argv[1]
    opath = sys.argv[2]
    filenames = [nm for nm in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, nm))]
    oname:str = 'coincidence'
    m = re.search('\w+.(tmo\w+).(r_\d+).h5$',filenames[0])
    if m:
        oname += '.%s.%s.h5'%(m.group(1),m.group(2))
    hist = np.zeros((1<<12,1<<12),dtype=np.uint64)
    h = {}
    tofs = {}
    addresses = {}
    nedges = {}
    maxtof:np.uint32 = 1<<15
    t0=11000
    #scale = 1
    corrports:List(str) = []
    sumens:np.uint64 = 0
    encalib:float = 1.0

    for name in filenames[:10]:
        print(name)
        with h5py.File(os.path.join(inpath,name),'r') as f:
            runstr = [k for k in f.keys()][0]
            detstr = 'mrco_hsd'
            hsd = f[runstr][detstr]
            fzp = f[runstr]['tmo_fzppiranha']
            xgmd = f[runstr]['xgmd']
            ports = [p for p in hsd.keys()]
            ens = xgmd['energy'][()]
            encalib = np.uint32(xgmd['energy'].attrs['scale'])
            sumens += sum(ens)
            cents = fzp['centroids'][()]
        
            corrports = [ports[int(i)] for i in sys.argv[3:5]]

            h.update({corrports[0]:[int(0)]*(hist.shape[0])})
            h.update({corrports[1]:[int(0)]*(hist.shape[1])})

            for p in corrports:
                tofs.update({p:hsd[p]['tofs'][()]})
                addresses.update({p:hsd[p]['addresses'][()]})
                nedges.update({p:hsd[p]['nedges'][()]})
                for t in tofs[p]:
                    h[p][ max(0,min(len(h[p])-1, int(t-t0)))] +=1
            #scale = len(addresses[corrports[0]])
            for i in range(len(addresses[corrports[0]])):
                n1 = nedges[corrports[0]][i]
                n2 = nedges[corrports[1]][i]
                a1 = addresses[corrports[0]][i]
                a2 = addresses[corrports[1]][i]
                for t1 in tofs[corrports[0]][a1:a1+n1]:
                    for t2 in tofs[corrports[1]][a2:a2+n2]:
                        ind1 = max(0,min(hist.shape[0]-1, int(t1-t0)))
                        ind2 = max(0,min(hist.shape[1]-1, int(t2-t0)))
                        hist[ind1,ind2] += 1
            #falsecoince = np.outer(h[corrports[0]],h[corrports[1]]).astype(float)
            #falsecoince /= float(scale>>4)


    with h5py.File(os.path.join(opath,oname),'a') as o:
        ds = o.create_dataset('%s.%s'%([str(s) for s in corrports]),data=hist,dtype=np.uint32)
        ds.attrs.create('t0',data = t0,dtype=np.uint32)
        ds.attrs.create('totalJ',data = sumens//(encalib),dtype=np.uint64)

    return


if __name__ == "__main__":
    if len(sys.argv)>4:
        main()
    else:
        print('point me to an input directory, and then a different output directory, and then the two ports you want me to correlate')


