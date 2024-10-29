import numpy as np
import typing
from typing import List
import h5py
import sys
import math
import statistics as stats

def quickCentroid(data):
    sz = len(data)
    base:np.uint32 = stats.mode(data)
    num:np.uint32 = sum([i*d for d in data]) - base*((sz)*(sz-1))>>1
    den:np.uint32 = sum(data) - base*sz
    return (np.uint16(num//den),np.uint32(num))

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
    def __init__(self,thresh=0) -> None:
        self.v = []
        self.vsize = int(0)
        self.vc = []
        self.vs = []
        self.vw = []
        self.initState = True
        self.vlsthresh = thresh
        self.winstart = 0
        self.winstop = 1<<11
        self.processAlgo = 'piranha'
        return

    @classmethod
    def slim_update_h5(cls,f,spect,vlsEvents):
        grpvls = None
        if 'spect' in f.keys():
            grpvls = f['spect']
        else:
            grpvls = f.create_group('spect')

        grpvls.create_dataset('centroids',data=spect.vc,dtype=np.uint16)
        grpvls.create_dataset('sum',data=spect.vs,dtype=np.uint64)
        grpvls.create_dataset('width',data=spect.vw,dtype=np.uint32)
        grpvls.attrs.create('size',data=spect.vsize,dtype=np.int32)
        grpvls.create_dataset('events',data=vlsEvents)
        return 

    @classmethod
    def update_h5(cls,f,spect,spcEvents):
        grpspc = None
        grprun = None
        for rkey in spect.keys():
            for name in spect[rkey].keys():
                thisspect = spect[rkey][name]
                rstr = spect[rkey][name].get_runstr()
                if rstr not in f.keys():
                    f.create_group(rstr)
                if name not in f[rstr].keys():
                    f.create_group('spect')
                grpspc = f[rstr][name]
                spcdata = grpspc.create_dataset('data',data=spect.v,dtype=int)
                grpspc.create_dataset('centroids',data=spect.vc,dtype=np.uint16)
                grpspc.create_dataset('sum',data=spect.vs,dtype=np.uint64)
                grpspc.create_dataset('width',data=spect.vw,dtype=np.uint16)
                grpspc.attrs.create('size',data=spect.vsize,dtype=np.int32)
                grpspc.create_dataset('events',data=spcEvents)
        return

    def setProcessAlgo(self,alg='pirnaha'):
        self.processAlgo = alg

    def get_runstr(self):
        return 'run_%04i'%self.runkey

    def get_runkey(self):
        return self.runkey

    def set_runkey(self,r:int):
        self.runkey = r
        return self

    def set_name(self,n:str):
        self.name = n
        return self

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


    def process(self,s):
        if self.processAlgo=='prianha':
            return self.process_piranha(s)
        else:
            return self.process_vls(s)


    def process_piranha(self,s):
        sz = len(s)
        base:np.uint32 = stats.mode(s)
        sumi2y:np.uint32 = sum([i*i*d for d in s]) - base*(n*(n-1)*(2*(n-1)+1))//6 #sum([i*i for i in range(sz)]) # for large sz this approaches sum([i**2 for i in range(sz)]) --> (sz**3)/3
        sumiy:np.uint32 = sum([i*d for d in s]) - base*(sz*(sz-1))>>1
        sumy:np.uint32 = sum(s) - base*sz
        c = sumiy/sumy
        w = math.sqrt(sumi2y)
        if self.initState:
            self.vsize = len(s)
            self.v = [v for v in s]
            self.vc = [np.uint16(c)]
            self.vs = [np.uint64(s)]
            self.vw = [np.uint16(w)]
            self.initState = False
        else:
            self.v += [v for v in s]
            self.vc += [np.uint16(c)]
            self.vs += [np.uint64(s)]
            self.vw += [np.uint16(w)]

        return True

    def process_vls(self, wv):
        mean = np.int16(np.mean(wv[1800:])) # this subtracts baseline
        if (np.max(wv)-mean)<self.vlsthresh:
            return False
        d = np.copy(wv-mean).astype(np.int16)
        c,s = getCentroid(d[self.winstart:self.winstop],pct=0.8)
        if self.initState:
            self.v = [d]
            self.vsize = len(self.v)
            self.vc = [np.uint16(c)]
            self.vs = [np.uint64(s)]
            self.vw = [np.uint16(self.winstop-self.winstart)]
            self.initState = False
        else:
            self.v += [d]
            self.vc += [np.uint16(c)]
            self.vs += [np.uint64(s)]
            self.vw += [np.uint16(self.winstop-self.winstart)]
        return True

    def set_initState(self,state: bool):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self

