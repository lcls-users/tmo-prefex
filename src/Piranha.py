import numpy as np
import typing
from typing import List
import h5py
import sys
import math
import statistics as stats

def getCentroid(data,pct=.8):
    csum = np.cumsum(data.astype(float))
    s = float(csum[-1])*pct
    csum /= csum[-1]
    inds = np.where((csum>(.5-pct/2.))*(csum<(.5+pct/2.)))
    tmp = np.zeros(data.shape,dtype=float)
    tmp[inds] = data[inds].astype(float)
    num = np.sum(tmp*np.arange(data.shape[0],dtype=float))
    return (num/s,np.uint64(s))

def shouldSplit(data):
    inds = np.arange(data.shape[0])
    maxind = np.argmax(data)
    com = int(np.sum(data.astype(float)*inds)/np.sum(data))
    return bool(abs(com-maxind)>(data.shape[0]>>3))

def getSplit(data):
    index = data.shape[0]-(data.shape[0]>>3)
    csum = np.cumsum(data)
    while csum[index]>(csum[-1]>>1):
        index -= index>>3
        #print(index)
    return index

class Spect:
    def __init__(self,thresh) -> None:
        self.v = []
        self.vsize = int(0)
        self.vc = []
        self.vs = []
        self.initState = True
        self.vlsthresh = thresh
        self.winstart = 0
        self.winstop = 1<<11
        return

    @classmethod
    def slim_update_h5(cls,f,spect,vlsEvents):
        grpvls = None
        if 'spect' in f.keys():
            grpvls = f['spect']
        else:
            grpvls = f.create_group('spect')

        grpvls.create_dataset('centroids',data=spect.vc,dtype=np.float16)
        grpvls.create_dataset('sum',data=spect.vs,dtype=np.uint64)
        grpvls.attrs.create('size',data=spect.vsize,dtype=np.int32)
        grpvls.create_dataset('events',data=vlsEvents)
        return 

    @classmethod
    def update_h5(cls,f,spect,vlsEvents):
        grpvls = None
        if 'spect' in f.keys():
            grpvls = f['spect']
        else:
            grpvls = f.create_group('spect')

        grpvls.create_dataset('data',data=spect.v,dtype=int)
        grpvls.create_dataset('centroids',data=spect.vc,dtype=np.float16)
        grpvls.create_dataset('sum',data=spect.vs,dtype=np.uint64)
        grpvls.attrs.create('size',data=spect.vsize,dtype=np.int32)
        grpvls.create_dataset('events',data=vlsEvents)
        return

    def setthresh(self,x):
        self.vlsthresh = x
        return self

    def setwin(self,low,high):
        self.winstart = int(low)
        self.winstop = int(high)
        return self

    def test(self,wv):
        mean = np.int16(0)
        if type(wv)==type(None):
            return False
        try:
            mean = np.int16(np.mean(wv[1800:])) # this subtracts baseline
        except:
            print('Damnit, Piranha!')
            return False
        else:
            if (np.max(wv)-mean)<self.vlsthresh:
                #print('Minnow, not a Piranha!')
                return False
        return True

    def process(self, wv):
        mean = np.int16(np.mean(wv[1800:])) # this subtracts baseline
        if (np.max(wv)-mean)<self.vlsthresh:
            return False
        d = np.copy(wv-mean).astype(np.int16)
        c,s = getCentroid(d[self.winstart:self.winstop],pct=0.8)
        if self.initState:
            self.v = [d]
            self.vsize = len(self.v)
            self.vc = [np.float16(c)]
            self.vs = [np.uint64(s)]
            self.initState = False
        else:
            self.v += [d]
            self.vc += [np.float16(c)]
            self.vs += [np.uint64(s)]
        return True

    def set_initState(self,state: bool):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self

