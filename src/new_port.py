""" Yup, I'm a new port.  Unlike the old port, I
only store static configuration data.
If you have per-event information, talk to WaveData or FexData
about it.
"""

import numpy as np
from pydantic import BaseModel

# Parse: cfg = PortConfig.model_validate_json('{"chankey":1, ...}')
# Serialize: cfg.model_dump_json()
class PortConfig(BaseModel):
    id: int
    chan: int
    is_fex: bool
    hsdname: str
    inflate: int = 1
    expand: int = 1
    logic_thresh: int = -1*(1<<20)
    roll_on: int = 256
    roll_off: int = 256
    nadcs: int = 4
    t0: int = 0
    baselim: int = 1<<6

class PortData:
    def to_h5(self):
        pass

class WaveData(PortData):
    cfg: PortConfig

    def __init__( self
                , cfg: PortConfig
                , wave = None
                ) -> None:
        self.cfg = cfg
        assert not cfg.is_fex
        #self.processAlgo = 'wave'

        self.wave = None
        if wave is not None:
            self.wave = wave[self.cfg.id][0]
        self.ok = self.wave is not None

    def set_baseline(self, val: np.ndarray) -> "WaveData":
        self.baseline = np.uint32(val)
        return self

    def process(self,slist,xlist=[0]) -> bool:
        cfg = self.cfg
        e:List[np.int32] = []
        de = []
        ne = 0
        r = []
        s = slist[0]
        x = xlist[0]
        if s is None:
            #self.addsample(np.zeros((2,),np.int16),np.zeros((2,),np.float16))
            e:List[np.int32] = []
            de = []
            ne = 0
            return False
        else:
            if len(self.addresses)%100==0:
                r = np.copy(s).astype(np.uint16)
            for adc in range(cfg.nadcs): # correcting systematic baseline differences for the four ADCs.
                b = np.mean(s[adc:cfg.baselim+adc:cfg.nadcs])
                s[adc::cfg.nadcs] = (s[adc::cfg.nadcs] ) - np.int32(b)
            #logic = fftLogic(s,inflate=cfg.inflate,nrolloff=cfg.nrolloff) #produce the "logic vector"
            logic = fftLogic_f16(s,inflate=cfg.inflate,nrolloff=cfg.nrolloff) #produce the "logic vector"
            e,de,ne = self.scanedges_simple(logic) # scan the logic vector for hits
        self.e = e
        self.de = de
        self.ne = ne

        self.addresses = [np.uint64(0)]
        self.nedges = [np.uint64(ne)]
        if ne>0:
            self.tofs += self.e
            self.slopes += self.de
        # appending
        #   self.addresses += [np.uint64(len(self.tofs))]
        #   self.nedges += [np.uint64(ne)]
        #   if ne>0:
        #       self.tofs += self.e
        #       self.slopes += self.de
        if len(self.addresses)%100==0:
            self.addsample(r,s,logic)
        return True

class FexData(PortData):
    cfg: PortConfig

    def __init__( self
                , cfg: PortConfig
                , peak = None
                ) -> None:
        self.cfg = cfg
        assert cfg.is_fex
        self.processAlgo = 'fex2hits'

        self.peak = None
        if peak is not None:
            self.peak = peak[self.cfg.id][0]
        self.ok = self.peak is not None

    def set_baseline(self, val: np.ndarray) -> "PortData":
        self.baseline = np.uint32(val)
        return self

    def process(self,s,x=0) -> bool:
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
        r = []
        goodlist = [s is not None for s in slist]
        if not all(goodlist)
            print(goodlist) 
            return False
        else:
            for i,s in enumerate(slist):
                if len(self.addresses)%100==0:
                    self.r = list(s.astype(np.int16, copy=True))
                ## no longer needing to correct for the adc offsets. ##
                logic = fftLogic_fex(s,inflate=cfg.inflate,nrollon=cfg.roll_on,nrolloff=cfg.roll_off) #produce the "logic vector"
                #e,de,ne = self.scanedges_simple(logic) # scan the logic vector for hits
                if len(self.addresses)%4==0:
                    self.addsample(r,s,logic)

                self.e += e
                self.de += de
                self.ne += ne

        self.addresses = [np.uint64(0)]
        self.nedges = [np.uint64(ne)]
        if ne>0:
            self.tofs += self.e
            self.slopes += self.de

        return True
