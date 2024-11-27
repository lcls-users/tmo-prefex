import numpy as np
import typing
from typing import List
import h5py
import sys
import math
import statistics as stats
from utils import getCentroid
import re

def quickCentroid(data):
    sz = len(data)
    base:np.uint32 = stats.mode(data)
    num:np.uint32 = sum([i*d for d in data]) - base*((sz)*(sz-1))>>1
    den:np.uint32 = sum(data) - base*sz
    return (np.uint16(num//den),np.uint32(num))

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
    def __init__(self,thresh=(1<<3)) -> None:
        self.v = []
        self.vsize = int(0)
        self.vc = []
        self.vs = []
        self.vw = []
        self.vstarts = []
        self.vlens = []
        self.initState = True
        self.thresh = thresh
        self.winstart = 0
        self.winstop = 1<<11
        self.processAlgo = 'piranha_slim' # 'piranha'
        return
    
    def reset(self):
        self.vc.clear()
        self.vs.clear()
        self.vw.clear()
        self.vstarts.clear()
        self.vlens.clear()
        self.v.clear()
        self.vsize = int(0)

    @classmethod
    def slim_update_h5(cls,f,spect,vlsEvents):
        print('slim_update() not yet implemented')
        '''
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
        '''
        return 

    @classmethod
    def update_h5(cls,f,spect,spcEvents):
        grpspc = None
        grprun = None
        for rkey in spect.keys():
            for name in spect[rkey].keys():
                thisspect = spect[rkey][name]
                rstr = spect[rkey][name].get_runstr()
                if name not in f.keys():
                    f.create_group(name)
                grpspc = f[name]
                grpspc.attrs.create('run',data=rstr)
                grpspc.create_dataset('data',data=spect[rkey][name].v,dtype=np.uint16)
                grpspc.create_dataset('starts',data=spect[rkey][name].vstarts,dtype=np.uint32)
                grpspc.create_dataset('lens',data=spect[rkey][name].vlens,dtype=np.uint16)
                grpspc.create_dataset('centroids',data=spect[rkey][name].vc,dtype=np.uint16)
                grpspc.create_dataset('sum',data=spect[rkey][name].vs,dtype=np.uint64)
                grpspc.create_dataset('widths',data=spect[rkey][name].vw,dtype=np.uint16)
                grpspc.attrs.create('size',data=spect[rkey][name].vsize,dtype=np.int32)
                grpspc.create_dataset('events',data=spcEvents)
        return

    def setProcessAlgo(self,alg='piranha'):
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
        self.thresh = x
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
            if (np.max(wv)-mean)<self.thresh:
                #print('Minnow, not a Piranha!')
                return False
        return True


    def process(self,s):
        if re.search('piranha',self.processAlgo):
            if re.search('slim',self.processAlgo):
                return self.process_piranha_slim(s)
            return self.process_piranha(s)
        else:
            return self.process_vls(s)


    def process_piranha_slim(self,signal):
        sz = len(signal)
        base:np.uint32 = int(stats.mode(signal>>2)<<2)
        sumiy:np.uint32 = sum([i*(int(d)-int(base)) for i,d in enumerate(signal)])# - base*(sz*(sz-1))>>1
        #sumiy:np.uint32 = sum([i*d for i,d in enumerate(s)]) - base*(sz*(sz-1))>>1
        sumy:np.uint32 = sum(signal) 
        if np.max(signal) < base<<1:
            if self.initState:
                self.vc = [np.uint16(0)]
                self.vs = [np.uint64(0)]
                self.vw = [np.uint16(0)]
                self.initState = False
            else:
                self.vc += [np.uint16(0)]
                self.vs += [np.uint64(0)]
                self.vw += [np.uint16(0)]
            return True

        sumy -= base*sz
        cent:np.uint16 = 0 # sumiy//sumy
        sumi2y:np.int32 = 0
        if sumy>0:
            cent:np.uint16 = sumiy//sumy
            sumi2y = sum([(abs(int(i)-int(cent))**2)*(d-base) for i,d in enumerate(signal) if d>(base+self.thresh)])# - base*(n*(n-1)*(2*(n-1)+1))//6 #sum([i*i for i in range(sz)]) # for large sz this approaches sum([i**2 for i in range(sz)]) --> (sz**3)/3
            #sumi2y:np.uint32 = sum([(i-cent)**2*d for i,d in enumerate(s)]) - base*sz*(sz-1)*(2*sz-1)//6 #sum([i*i for i in range(sz)]) # for large sz this approaches sum([i**2 for i in range(sz)]) --> (sz**3)/3
        width:np.uint16 = np.uint16(0)
        if sumi2y>0 and sumy>0:
            width:np.uint16 = math.isqrt(sumi2y//sumy)

        if self.initState:
            self.vc = [np.uint16(cent)]
            self.vs = [np.uint64(sumy)]
            self.vw = [np.uint16(width)]
            self.initState = False
        else:
            self.vc += [np.uint16(cent)]
            self.vs += [np.uint64(sumy)]
            self.vw += [np.uint16(width)]
        return True

    def process_piranha(self,signal):
        sz = len(signal)
        base:np.uint32 = int(stats.mode(signal>>2)<<2)
        sumiy:np.uint32 = sum([i*(int(d)-int(base)) for i,d in enumerate(signal)])# - base*(sz*(sz-1))>>1
        #sumiy:np.uint32 = sum([i*d for i,d in enumerate(s)]) - base*(sz*(sz-1))>>1
        sumy:np.uint32 = sum(signal) 
        if np.max(signal) < base<<1:
            if self.initState:
                self.vstarts = [len(self.v)]
                self.vlens = [0]
                self.vc = [np.uint16(0)]
                self.vs = [np.uint64(0)]
                self.vw = [np.uint16(0)]
                self.initState = False
            else:
                self.v += []
                self.vstarts += [len(self.v)]
                self.vlens += [0]
                self.vc += [np.uint16(0)]
                self.vs += [np.uint64(0)]
                self.vw += [np.uint16(0)]
            return True

        sumy -= base*sz
        cent:np.uint16 = 0 # sumiy//sumy
        sumi2y:np.int32 = 0
        if sumy>0:
            cent:np.uint16 = sumiy//sumy
            sumi2y = sum([(abs(int(i)-int(cent))**2)*(d-base) for i,d in enumerate(signal) if d>(base+self.thresh)])# - base*(n*(n-1)*(2*(n-1)+1))//6 #sum([i*i for i in range(sz)]) # for large sz this approaches sum([i**2 for i in range(sz)]) --> (sz**3)/3
            #sumi2y:np.uint32 = sum([(i-cent)**2*d for i,d in enumerate(s)]) - base*sz*(sz-1)*(2*sz-1)//6 #sum([i*i for i in range(sz)]) # for large sz this approaches sum([i**2 for i in range(sz)]) --> (sz**3)/3
        width:np.uint16 = np.uint16(0)
        if sumi2y>0 and sumy>0:
            width:np.uint16 = math.isqrt(sumi2y//sumy)
        else:
            width = 1<<7
        #print(cent,width,cent-width,cent+width)
        self.winstart = cent - (width)
        self.winstop = cent + (width)
        if cent<width:
            self.winstart = 0
        if self.winstop > (sz-1):
            self.winstop = sz

        if self.initState:
            self.v = [v for v in signal[self.winstart:self.winstop]]
            self.vstarts = [len(self.v)]
            self.vlens = [self.winstop-self.winstart]
            self.vc = [np.uint16(cent)]
            self.vs = [np.uint64(sumy)]
            self.vw = [np.uint16(width)]
            self.initState = False
        else:
            self.v += [v for v in signal[self.winstart:self.winstop]]
            self.vstarts += [len(self.v)]
            self.vlens += [self.winstop-self.winstart]
            self.vc += [np.uint16(cent)]
            self.vs += [np.uint64(sumy)]
            self.vw += [np.uint16(width)]
        return True

    def process_vls(self, wv):
        mean = np.int16(np.mean(wv[1800:])) # this subtracts baseline
        if (np.max(wv)-mean)<self.thresh:
            return False
        d = np.copy(wv-mean).astype(np.int16)
        cent,sumy = getCentroid(d[self.winstart:self.winstop],pct=0.8)
        if self.initState:
            self.v = [v for v in d]
            self.vsize = len(self.v)
            self.vc = [np.uint16(c)]
            self.vs = [np.uint64(sumy)]
            self.vw = [np.uint16(self.winstop-self.winstart)]
            self.initState = False
        else:
            self.v += [v for v in d]
            self.vc += [np.uint16(c)]
            self.vs += [np.uint64(sumy)]
            self.vw += [np.uint16(self.winstop-self.winstart)]
        return True

    def set_initState(self,state: bool):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self

