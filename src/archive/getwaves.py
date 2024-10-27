#!/usr/bin/python3

import h5py
import numpy as np
import sys

def pkey(p):
    return 'port_%i'%p

def process_tofs(fname,t0s):
    ports = [0,1,4,5,12,13,14,15]
    bins = np.arange(2**18,dtype=float)
    log2bins = np.linspace(10,16,2**15)
    with h5py.File(fname,'r') as f:
        #t0s = {}
        data = {}
        nedges = {}
        for p in ports: 
            data.update( {'port_%i'%p:f['port_%i'%p]['tofs'][()]} ) 
            #t0s.update({'port_%i'%p:f['port_%i'%p].attrs['t0']})
            nedges.update( {'port_%i'%p:f['port_%i'%p]['nedges'][()]} )
            #h = np.histogram(data['port_%i'%p][()]-t0s['port_%i'%p]*fudge_scale,bins)[0]
            #h = np.histogram(data['port_%i'%p][()],bins)[0]
            h = np.histogram(data['port_%i'%p][()]-t0s[pkey(p)],bins)[0]
            np.savetxt('tmp_port_%i.dat'%p,np.column_stack((bins[:-1],h)),fmt = '%.2f')
            #h = np.histogram(np.log2(data['port_%i'%p][()]-t0s['port_%i'%p]*fudge_scale),log2bins)[0]
            h = np.histogram(np.log2(data['port_%i'%p][()]-t0s[pkey(p)]),log2bins)[0]
            np.savetxt('tmp_log2_port_%i.dat'%p,np.column_stack((log2bins[:-1],h)),fmt = '%.6f')
    return data,nedges

def process_waves(fname):
    ports = [0,1,4,5,12,13,14,15]
    with h5py.File(fname,'r') as f:
        waves = {}
        for p in ports:
            nwaves = 0
            for k in f['port_%i'%p]['waves'].keys(): 
                if nwaves>100:
                    continue
                waves.update( {'port_%i%s'%(p,k):f['port_%i'%p]['waves'][k][()]} )
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
        h = np.histogram(data['port_%i'%p]%1,bins)[0]
        np.savetxt('tmpmod_%i.dat'%p,np.column_stack((bins[:-1],h)),fmt='%.3f')
    
    return

if __name__ == '__main__':
    main()
