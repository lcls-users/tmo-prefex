""" Yup, I'm a new port.  Unlike the old port, I
only store static configuration data.
If you have per-event information, talk to WaveData or FexData
about it.
"""

from typing import List, Any
import time

import numpy as np
from pydantic import BaseModel

from Ports import cfdLogic, fftLogic_f16, fftLogic_fex, fftLogic
from utils import mypoly,tanhInt,tanhFloat,randomround

_rng = np.random.default_rng( time.time_ns()%(1<<8) )

# Parse: cfg = PortConfig.model_validate_json('{"chankey":1, ...}')
# Serialize: cfg.model_dump_json()
class PortConfig(BaseModel):
    id: int
    chankey: int # was hsd # was hsd
    is_fex: bool
    hsdname: str
    inflate: int = 1
    expand: int = 1
    logic_thresh: int = -1*(1<<20) # was logicthresh
    roll_on: int = 256
    roll_off: int = 256
    nadcs: int = 4
    t0: int = 0
    baselim: int = 1<<6

    def scanedges_stupid(self,d):
        tofs = []
        slopes = []
        sz = d.shape[0]
        i:int = int(0)
        while i < sz-1:
            while d[i] > self.logic_thresh:
                i += 1
                if i==sz-1: return tofs,slopes,len(tofs)
            while i<sz-1 and d[i]<0:
                i += 1
            stop = i
            ''' dx / (Dy) = dx2/dy2 ; dy2*dx/Dy - dx2 ; x2-dx2 = stop - dy2*1/Dy'''
            if d[stop]>(-1*d[stop-1]):
                stop -= 1
            i += 1
            tofs += [np.uint32(stop)] 
            slopes += [d[stop]-d[stop-1]] 
        return tofs,slopes,np.uint64(len(tofs))

    def scanedges_simple(self,d):
        tofs = []
        slopes = []
        sz = d.shape[0]
        i:int = int(10)
        while i < sz-10:
            while d[i] > self.logic_thresh:
                i += 1
                if i==sz-10: return tofs,slopes,len(tofs)
            while i<sz-10 and d[i]<0:
                i += 1
            stop = i
            ''' dx / (Dy) = dx2/dy2 ; dy2*dx/Dy - dx2 ; x2-dx2 = stop - dy2*1/Dy'''
            x0 = float(stop) - float(d[stop])/float(d[stop]-d[stop-1])
            i += 1
            v = float(self.expand)*float(x0)
            tofs += [np.uint32(randomround(v,_rng))] 
            slopes += [d[stop]-d[stop-1]] 
        return tofs,slopes,np.uint64(len(tofs))

    def scanedges(self,d):
        tofs = []
        slopes = []
        sz = d.shape[0]
        newtloops = 6
        order = 3 # this should stay fixed, since the logic zeros crossings really are cubic polys
        i = 10
        while i < sz-10:
            while d[i] > self.logic_thresh:
                i += 1
                if i==sz-10: return tofs,slopes,len(tofs)
            while i<sz-10 and d[i]<d[i-1]:
                i += 1
            start = i-1
            i += 1
            while i<sz-10 and d[i]>d[i-1]:
                i += 1
            stop = i
            i += 1
            if (stop-start)<(order+1):
                continue
            x = np.arange(stop-start,dtype=float) # set x to index values
            y = d[start:stop] # set y to vector values
            x0 = float(stop)/2. # set x0 to halfway point
            #y -= (y[0]+y[-1])/2. # subtract average (this gets rid of residual DC offsets)
    
            theta = np.linalg.pinv( mypoly(np.array(x).astype(float),order=order) ).dot(np.array(y).astype(float)) # fit a polynomial (order 3) to the points
            for j in range(newtloops): # 3 rounds of Newton-Raphson
                X0 = np.array([np.power(x0,int(k)) for k in range(order+1)])
                x0 -= theta.dot(X0)/theta.dot([i*X0[(k+1)%(order+1)] for k in range(order+1)]) # this seems like maybe it should be wrong
            tofs += [float(start + x0)] 
            #X0 = np.array([np.power(x0,int(i)) for k in range(order+1)])
            #slopes += [np.int64(theta.dot([i*X0[(i+1)%(order+1)] for i in range(order+1)]))]
            slopes += [float((theta[1]+x0*theta[2])/2**18)] ## scaling to reign in the obscene derivatives... probably shoul;d be scaling d here instead
        return tofs,slopes,np.uint32(len(tofs))


class PortData:
    cfg: PortConfig
    event: int
    ok: bool
    raw: np.ndarray # np.uint16
    logic: List[Any]
    tofs: List[Any]
    slopes: List[Any]
    nedges: np.uint64

class WaveData(PortData):
    def __init__( self
                , cfg: PortConfig
                , event: int
                , wave = None
                ) -> None:
        self.cfg = cfg
        self.event = event
        assert not cfg.is_fex
        #self.processAlgo = 'wave'

        self.wave = None
        if wave is not None:
            self.wave = wave[self.cfg.id][0]
        self.ok = self.wave is not None

    def setup(self) -> "WaveData": # was set_baseline
        # FIXME: just store as self.raw???
        self.slist = [ np.array(data.wave, dtype=np.int16) ] # presumably 12 bits unsigned input, cast as int16_t since will immediately in-place subtract baseline
        self.xlist = [0]
        #self.baseline = np.uint32(0)
        return self

    def process(self) -> bool:
        cfg = self.cfg
        e:List[np.int32] = []
        de = []
        ne = 0
        r = []
        s = self.slist[0] # data.wave as np.int16
        x = self.xlist[0] # == 0
        if s is None:
            return False
        
        for adc in range(cfg.nadcs): # correcting systematic baseline differences for the four ADCs.
            b = np.mean(s[adc:cfg.baselim+adc:cfg.nadcs])
            s[adc::cfg.nadcs] = (s[adc::cfg.nadcs] ) - np.int32(b)
        #logic = fftLogic(s,inflate=cfg.inflate,nrolloff=cfg.nrolloff) #produce the "logic vector"
        logic = fftLogic_f16(s,inflate=cfg.inflate,nrolloff=cfg.nrolloff) #produce the "logic vector"
        e,de,ne = cfg.scanedges_simple(logic) # scan the logic vector for hits
        self.raw = s.astype(np.uint16, copy=True)
        self.logic = logic
        self.tofs = e
        self.slopes = de
        self.nedges = np.uint64(ne)
        return True

class FexData(PortData):
    def __init__( self
                , cfg: PortConfig
                , event: int
                , peak = None
                ) -> None:
        self.cfg = cfg
        self.event = event
        assert cfg.is_fex
        self.processAlgo = 'fex2hits'

        self.peak = None
        if peak is not None:
            self.peak = peak[self.cfg.id][0]
        self.ok = self.peak is not None

    def setup(self) -> "FexData": # was set_baseline
        xlist: List[int] = []
        slist: List[np.ndarray] = []

        nwins = len(self.peak[0])
        baseline = 0
        if nwins > 2: # always reports the start of and the end of the fex active window.
            baseline = np.sum(self.peak[1][0].astype(np.uint32))
            baseline //= len(self.peak[1][0])
            xlist.extend(self.peak[0][1:nwins-2])
            for i in range(1,nwins-2):
                #xlist += [ self.peak[0][i] ]
                slist += [np.array(self.peak[1][i], dtype=np.int32)]

        self.xlist = xlist
        self.slist = slist
        self.baseline = np.uint32(baseline)
        return self

    def process(self) -> bool:
        s = self.slist
        x = self.xlist
        if self.processAlgo =='fex2coeffs':
            return self.process_fex2coeffs(s,x)
        elif self.processAlgo == 'fex2hits':
            return self.process_fex2hits(s,x)
        raise KeyError(self.processAlgo)

    def process_fex2coeffs(self,s,x):
        print('HERE HERE HERE HERE')
        return True

    def process_fex2hits(self,slist,xlist):
        cfg = self.cfg
        e = []
        de = []
        ne = 0
        goodlist = [s is not None for s in slist]
        if not all(goodlist):
            print(f"process_fex2hits: some bad samples, {goodlist}")
            return False
        for i,s in enumerate(slist):
            ## no longer needing to correct for the adc offsets. ##
            ## logic = fftLogic_fex(s,self.baseline,inflate=cfg.inflate,nrollon=cfg.roll_on,nrolloff=cfg.roll_off) #produce the "logic vector"
            logic = cfdLogic(s)
            es,des,nes = cfg.scanedges_stupid(logic) # scan the logic vector for hits

            e.extend( es )
            de.extend( des )
            ne += nes

        if len(slist) > 0:
            self.raw = s.astype(np.uint16, copy=True)
            self.logic = logic
        else:
            self.raw = np.zeros(0, dtype=np.uint16)
            self.logic = np.zeros(0, dtype=np.uint16)
        self.tofs = e
        self.slopes = de
        self.nedges = np.uint64(ne)
        return True
