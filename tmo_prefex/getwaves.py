#!/usr/bin/python3

import h5py
import numpy as np
import sys

def pkey(p):
    return '%i'%p

def process_tofs(fname,t0s):
    ports = [0,1,4,5,12,13,14,15]
    bins = np.arange(2**18,dtype=float)
    log2bins = np.linspace(10,16,2**15)
    with h5py.File(fname,'r') as f:
        #t0s = {}
        data = {}
        nedges = {}
        for p in ports: 
            key = pkey(p)
            g = f[key]
            data.update( {key: g['tofs'][()]} ) 
            #t0s.update({key%p:g.attrs['t0']})
            nedges.update( {key: g['nedges'][()]} )
            d = data[key]
            #h = np.histogram(d[()]-t0s[key]*fudge_scale,bins)[0]
            #h = np.histogram(d[()],bins)[0]
            h = np.histogram(d[()]-t0s[key],bins)[0]
            np.savetxt('tmp_port_%i.dat'%p,np.column_stack((bins[:-1],h)),fmt = '%.2f')
            #h = np.histogram(np.log2(d[()]-t0s[key]*fudge_scale),log2bins)[0]
            h = np.histogram(np.log2(d[()]-t0s[key]),log2bins)[0]
            np.savetxt('tmp_log2_port_%i.dat'%p,np.column_stack((log2bins[:-1],h)),fmt = '%.6f')
    return data,nedges

def process_waves(fname):
    ports = [0,1,4,5,12,13,14,15]
    with h5py.File(fname,'r') as f1:
        #print(list(f.keys()))
        waves = {}
        for p in ports:
            #print(key, list(f[key].keys()))
            key = pkey(p)
            nwaves = 0
            for k in f[key]['waves'].keys(): 
                if nwaves>100:
                    continue
                waves.update( {(key,k):f[key]['waves'][k][()]} )
                nwaves += 1
    return waves



def main():
    if len(sys.argv)<2:
        print('give me an h5 filename')
        return
    print('overriding t0s with:')
    t0s = {pkey(0):73227,pkey(1):66973,pkey(2):66545,pkey(4):64796,pkey(5):66054,pkey(12):65712,pkey(13):65777,pkey(14):66891,pkey(15):71312,pkey(16):64887} # final, based on inflate=4 nr_expand=4
    _ = [print('%s\t%.1f'%(p,t0s[p])) for p in t0s.keys()]

    fname = sys.argv[1]
    waves = process_waves(fname)
    out = np.column_stack([waves[key][()] for key in waves.keys()])
    np.savetxt('tmpwaves.dat',out,fmt='%i')
    data,nedges = process_tofs(fname,t0s)
    bins=np.arange(2**8,dtype=float)/2.**8
    h=np.zeros(bins.shape[0]-1,dtype=int)
    for p in [0,1,4,5,12,13,14,15]:
        h = np.histogram(data[pkey(p)]%1,bins)[0]
        np.savetxt('tmpmod_%i.dat'%p,np.column_stack((bins[:-1],h)),fmt='%.3f')
    
    return

if __name__ == '__main__':
    main()
