""" Hsd-s (formerly referred to Port-s)
store configuration data for individual ports of an HSD.

Per-event data from an Hsd is stored in
WaveData or FexData -- depending on the HsdConfig.fex flag.
"""

from typing import List, Any, Union, Dict, Optional
import time
from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel
from stream import stream

from .utils import (
    mypoly, tanhInt, tanhFloat,
    randomround, quick_mean, concat,
    cfdLogic, fftLogic_f16, fftLogic_fex, fftLogic,
    calc_offsets
)

_rng = np.random.default_rng( time.time_ns()%(1<<8) )

# Parse: cfg = HsdConfig.model_validate_json('{"chankey":1, ...}')
# Serialize: cfg.model_dump_json()
class HsdConfig(BaseModel):
    id: int
    chankey: int # was hsd # was hsd
    is_fex: bool
    name: str
    inflate: int = 1 # inflate pads the DCT(FFT) with zeros, artificially over sampling the waveform
    expand: int = 1  # expand controls the fractional resolution for scanedges by scaling index values and then zero crossing round to intermediate integers.
    logic_thresh: int = -1*(1<<10) # was logicthresh
    roll_on: int = 256
    roll_off: int = 256
    nadcs: int = 4
    t0: int = 0
    baselim: int = 1<<6
    # size: int = p[key].sz*p[key].inflate ### need to also multiply by expand #### HERE HERE HERE HERE
    rate: float = 6.0e9 # digitizer sampling rate (Hz)

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

default_wave = HsdConfig(
    id=0, # psana's name for detector
    chankey=0, # our name for detector
    # -- may want to use chankey to store the PCIe
    #    address id or the HSD serial number.
    is_fex = False,
    name = '',
    inflate = 2,
    expand = 4,
    roll_on = 1<<6,
    roll_off = 1<<6,
)
default_fex = HsdConfig(
    id=0, # psana's name for detector
    chankey=0, # our name for detector
    # -- may want to use chankey to store the PCIe
    #    address id or the HSD serial number.
    is_fex = True,
    name = '',
    inflate = 2,
    expand = 4,
)


# FIXME: do we need .raw here?
# can the keys() be ordered into a contiguous range?
def get_portnums(hsd) -> Dict[int,int]:
    ports = list(hsd.raw._seg_configs().keys())
    ports.sort()
    return dict(enumerate(ports))


def setup_hsds(run, params, default=default_fex) -> Dict[str,Any]:
    """ Gather the dict of hsdname: hsd
    for all detectors ending in 'hsd'.

    Also adds defaults for this detector to params
    if not present.
    """
    hsdnames = [s for s in run.detnames if s.endswith('hsd')]

    hsds = {}
    for hsdname in hsdnames:
        hsd = run.Detector(hsdname)
        if hsd is None:
            print(f'run.Detector({hsdname}) is None!')
            continue

        hsds[hsdname] = hsd
        for i,k in get_portnums(hsd).items():
            idx = (hsdname,k)
            if idx not in params:
                cfg = default.copy()
                cfg.name = hsdname
                cfg.id = k
                cfg.chankey = i
                if default.is_fex:
                    # guessing that 3/4 of the pre and post extension for
                    # threshold crossing in fex is a good range for the
                    # roll on and off of the signal
                    cfg.roll_on = ( 3*int(hsd.raw._seg_configs()[k]
                                        .config.user.fex.xpre) )>>2
                    cfg.roll_off = (3*int(hsd.raw._seg_configs()[k]
                                        .config.user.fex.xpost) )>>2
                params[idx] = cfg
    return hsds

class HsdData:
    cfg: HsdConfig
    event: int
    ok: bool
    raw: np.ndarray # np.uint16
    logic: List[Any]
    tofs: List[Any]
    slopes: List[Any]
    nedges: np.uint32

class WaveData(HsdData):
    def __init__( self
                , cfg: HsdConfig
                , event: int
                , wave = None
                ) -> None:
        self.cfg = cfg
        self.event = event
        assert not cfg.is_fex
        #self.processAlgo = 'wave'

        self.ok = wave is not None
        if wave is None:
            return

        self.raw = np.array(wave, dtype=np.int16) # presumably 12 bits unsigned input, cast as int16_t since will immediately in-place subtract baseline
        #self.baseline = np.uint32(1<<8)

    def process(self):
        cfg = self.cfg
        s = self.raw # data.wave as np.int16
        x = 0
        
        for adc in range(cfg.nadcs): # correcting systematic baseline differences for the four ADCs.
            b = np.mean(s[adc:cfg.baselim+adc:cfg.nadcs])
            s[adc::cfg.nadcs] = (s[adc::cfg.nadcs] ) - np.int32(b)
        #logic = fftLogic(s,inflate=cfg.inflate,nrolloff=cfg.nrolloff) #produce the "logic vector"
        logic = fftLogic_f16(s,inflate=cfg.inflate,nrolloff=cfg.nrolloff) #produce the "logic vector"
        e,de,ne = cfg.scanedges_simple(logic) # scan the logic vector for hits
        #self.raw = s.astype(np.uint16, copy=True)
        self.logic = logic
        self.tofs = e
        self.slopes = de
        self.nedges = np.uint32(ne)
        return self

"""
TODO: ensure that process() logic follows this pattern:

           xlist += [ hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][0][i] ]
           slist += [ np.array(hsds[rkey][hsdname].raw.peaks(evt)[ key ][0][1][i],dtype=np.int32) ]
       elif eventnum > 100 and hsds[rkey][hsdname].raw.waveforms(evt) is not None:
           slist += [ np.array(hsds[rkey][hsdname].raw.waveforms(evt)[ key ][0] , dtype=np.int16) ] 
           xlist += [0]

           wv = hsds[hkey][hsdname].raw.waveforms(evt)[ key ][0]
           wvx = np.arange(wv.shape[0])
           y = [hsd.raw.peaks(evt)[1][0][1][i] for i in range(len(hsd.raw.peaks(evt)[1][0][1]))]
           x = [np.arange(hsd.raw.peaks(evt)[1][0][0][i],hsd.raw.peaks(evt)[1][0][0][i]+len(hsd.raw.peaks(evt)[1][0][1][i])) for i in range(len(y))]
           plt.plot(wv)
           _=[plt.plot(x[i],y[i]) for i in range(len(y))]
           plt.show()
       port[rkey][hsdname][key].process(slist,xlist) # this making a list out of the waveforms is to accommodate both the fex and the non-fex with the same Hsd object and .process() method.
"""
class FexData(HsdData):
    baseline: np.uint32

    def __init__( self
                , cfg: HsdConfig
                , event: int
                , peak = None
                ) -> None:
        self.cfg = cfg
        self.event = event
        assert cfg.is_fex
        self.processAlgo = 'fex2hits'

        if peak is not None:
            self.ok = self._setup(peak)
        else:
            self.ok = False

    def _setup(self, peak) -> "FexData":
        "compute and set self.{baseline, xlist, slist}"
        xlist: List[int] = []
        slist: List[np.ndarray] = []

        nwins = len(peak[0])
        baseline = 0
        if nwins > 2: # always reports the start of and the end of the fex active window.
            #baseline = np.sum(peak[1][0].astype(np.uint32))
            #baseline //= len(peak[1][0])
            baseline = peak[1][0].mean() # avoid int. overflow
            xlist.extend(peak[0][1:nwins-2])
            for i in range(1,nwins-2):
                #xlist += [ peak[0][i] ] # used extend, above.
                slist += [np.array(peak[1][i], dtype=np.int32)]

        self.xlist = xlist
        self.slist = slist
        self.baseline = np.uint32(baseline)

        goodlist = [s is not None for s in slist]
        if not all(goodlist):
            print(f"process_fex2hits: some bad samples, {goodlist}")
            return False
        return True

    def process(self):
        s = self.slist
        x = self.xlist
        if self.processAlgo =='fex2coeffs':
            self.process_fex2coeffs(s, x)
        elif self.processAlgo == 'fex2hits':
            self.process_fex2hits(s, x)
        else:
            raise KeyError(self.processAlgo)
        return self

    def process_fex2coeffs(self,s,x):
        print('HERE HERE HERE HERE')
        return True

    def process_fex2hits(self,slist,xlist):
        cfg = self.cfg
        e = []
        de = []

        if len(slist) > 2:
            for i,s in enumerate(slist[:-1]):
                if i == 0:
                    #self.baseline = quick_mean(s,4) # uint32
                    continue
                ## no longer needing to correct for the adc offsets. ##
                ## logic = fftLogic_fex(s,self.baseline,inflate=cfg.inflate,nrollon=cfg.roll_on,nrolloff=cfg.roll_off) #produce the "logic vector"
                # scan the logic vector for hits
                es,des,_ = cfdLogic(s,thresh=int(-512),offset=2)
                start = xlist[i]
                e.extend([start+v for v in es])
                de.extend( list(des) )

            self.raw = s.astype(np.uint16, copy=True)
            self.logic = np.zeros(0, dtype=np.uint16)
        else:
            self.raw = np.zeros(0, dtype=np.uint16)
            self.logic = np.zeros(0, dtype=np.uint16)
        self.tofs = e
        self.slopes = de
        self.nedges = np.uint16(len(e))
        return True

@stream
def run_hsds(events, hsds, params,
            ) -> Iterator[Optional[Dict[str,Any]]]:
    # assume runhsd is True if this fn. is called

    for eventnum, evt in events:
        out = {}

        ## test hsds
        completeEvent = True
        for hsdname, hsd in hsds.items():
          for key in get_portnums(hsd).values():
            idx = (hsdname, key)
            port = params[idx]
            if port.is_fex:
                peak = hsd.raw.peaks(evt)
                if peak is None:
                    print('%i: hsds[%s].raw.peaks(evt) is None'%(eventnum,repr(idx)))
                    completeEvent = False
                else:
                    out[idx] = FexData(port,
                                       eventnum,
                                       peak=peak[key][0])
                    completeEvent = out[idx].ok
            else:
                wave = hsd.raw.waveforms(evt)
                if wave is None:
                    print('%i: hsds[%s].raw.waveforms(evt) is None'%(eventnum,repr(idx)))
                    completeEvent = False
                else:
                    out[idx] = WaveData(port,
                                        eventnum,
                                        wave=wave[key][0])
                    completeEvent = out[idx].ok
            if not completeEvent:
                break

        if completeEvent:
            yield out
        else:
            yield None


def should_save_raw(eventnum):
    # first 10 of every 10, then first 10 of every 100, ...
    mod = 10
    cap = 100
    while eventnum > cap:
        mod *= 10
        cap *= 10
        if cap == 100000:
            break
    return (eventnum % mod) < 10

def save_hsd(waves: Union[List[WaveData], List[FexData]]
            ) -> Dict[str,Any]:
    """ Save a batch of data.

    Gathers up relevant information from the processed data
    of an hsd-type detector.
    """
    if len(waves) == 0:
        return {}
    events = [x.event for x in waves]
    nedges = [x.nedges for x in waves]
    events = []
    nedges = []
    tofs = []
    slopes = []
    for x in waves:
        if x.nedges == 0:
            continue
        events.append(x.event)
        nedges.append(x.nedges)
        tofs.append(x.tofs)
        slopes.append(x.slopes)

    # combine [raw,logic] together
    # at ea. rl_event[i], rl_addresses[i]
    # and identify raw_len[i] and logic_len[i] separately
    rl_idx = []
    rl_events = []
    rl_addresses = []
    raw_lens = []
    logic_lens = []
    k = 0
    for i, x in enumerate(waves):
        ev = x.event
        if should_save_raw(ev):
            u = len(waves[i].raw)
            v = len(waves[i].logic)
            if u+v == 0:
                continue
            rl_idx.append(i)
            rl_events.append(ev)
            rl_addresses.append(k)
            raw_lens.append(u)
            logic_lens.append(v)
            k += u+v

    nedges = np.array(nedges, dtype=np.uint64)
    if len(nedges) == 0:
        addresses = np.array([], dtype=np.uint64)
    else:
        addresses = calc_offsets(nedges)
    return dict(
        config = waves[0].cfg,
        events = np.array(events, dtype=np.uint32),
        addresses = addresses,
        tofs = concat(tofs),
        slopes = concat(slopes),
        nedges = nedges,

        rl_events = rl_events,
        rl_addresses = rl_addresses,
        raw_lens = raw_lens,
        logic_lens = logic_lens,
        rl_data = concat(concat([waves[i].raw,waves[i].logic])
                              for i in rl_idx ),
        #waves = waves[i].raw,
    )
