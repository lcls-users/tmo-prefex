import numpy as np
import typing
from typing import List
import h5py
import sys
import math
import statistics as stats
from utils import getCentroid
import re

def quickEdge(data):
    sz = len(data)
    base:np.uint32 = stats.mode(data)
    num:np.uint32 = sum([i*d for d in data]) - base*((sz)*(sz-1))>>1
    den:np.uint32 = sum(data) - base*sz
    return (np.uint16(num//den),np.uint32(num))

class Atm:
    def __init__(self,thresh=(1<<3)) -> None:
        self.v:List(np.int16) = []
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
        self.processAlgo = 'piranha_edge' # 'piranha'
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
    def slim_update_h5(cls,f,atm,vlsEvents):
        print('slim_update() not yet implemented')
        '''
        grpvls = None
        if 'atm' in f.keys():
            grpvls = f['atm']
        else:
            grpvls = f.create_group('atm')

        grpvls.create_dataset('centroids',data=atm.vc,dtype=np.uint16)
        grpvls.create_dataset('sum',data=atm.vs,dtype=np.uint64)
        grpvls.create_dataset('width',data=atm.vw,dtype=np.uint32)
        grpvls.attrs.create('size',data=atm.vsize,dtype=np.int32)
        grpvls.create_dataset('events',data=vlsEvents)
        '''
        return 

    @classmethod
    def update_h5(cls,f,atm,spcEvents):
        grpspc = None
        grprun = None
        for rkey in atm.keys():
            for name in atm[rkey].keys():
                thisatm = atm[rkey][name]
                rstr = atm[rkey][name].get_runstr()
                if rstr not in f.keys():
                    f.create_group(rstr)
                if name not in f[rstr].keys():
                    f[rstr].create_group(name)
                grpspc = f[rstr][name]
                grpspc.create_dataset('data',data=atm[rkey][name].v,dtype=np.uint16)
                grpspc.create_dataset('starts',data=atm[rkey][name].vstarts,dtype=np.uint32)
                grpspc.create_dataset('lens',data=atm[rkey][name].vlens,dtype=np.uint16)
                grpspc.create_dataset('centroids',data=atm[rkey][name].vc,dtype=np.uint16)
                grpspc.create_dataset('sum',data=atm[rkey][name].vs,dtype=np.uint64)
                grpspc.create_dataset('widths',data=atm[rkey][name].vw,dtype=np.uint16)
                grpspc.attrs.create('size',data=atm[rkey][name].vsize,dtype=np.int32)
                grpspc.create_dataset('events',data=spcEvents)
        return

    def setProcessAlgo(self,alg='piranha_edge'):
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
            print('Damnit, ATM Piranha!')
            return False
        else:
            if (np.max(wv)-mean)<self.thresh:
                #print('Minnow, not a Piranha!')
                return False
        return True


    def updateref(self,s):
        if s is not None and len(self.refatm)==len(s):
            self.refatm *= 3
            self.refatm += s
            self.refatm >>= 2
        else:
            self.refatm = s
        return True

    def process(self,s):
        if re.search('piranha',self.processAlgo):
            if re.search('edge',self.processAlgo):
                return self.process_piranha_edge(s)
            return self.process_piranha(s)
        else:
            return self.process_vls(s)


    def process_piranha_edge(self,signal):
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

    def set_initState(self,state: bool):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self

