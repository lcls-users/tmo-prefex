from typing import List, Any, Dict
import sys

from pydantic import BaseModel

import numpy as np

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

class SpectConfig(BaseModel):
    name: str
    vlsthresh: int
    winstart: int = 0
    winstop: int = 1<<11

def setup_spects(run, param):
    spectnames = [s for s in run.detnames \
                    if s.endswith('piranha')]

    dets = {}
    for name in names:
        det = run.Detector(name)
        if dat is None:
            print(f'run.Detector({name}) is None!')
            continue
        dets[name] = det

        idx = (name, 0)
        if idx not in params:
            cfg = SpectConfig(
                name = name,
                vlsthresh = 1000,
                winstart = 1024,
                winstop = 2048
            )
            params[idx] = cfg

    return dets

class SpectData:
    def __init__(self,
                 cfg: SpectConfig,
                 event: int,
                 wv
                 ) -> None:
        self.cfg = cfg
        self.event = event

        self.ok = False
        if wv is None:
            return
        try:
            # this subtracts baseline
            mean = np.int16(wv[1800:].mean())
        except:
            print('Damnit, Piranha!')
            return
        if (wv.max() - mean) < cfg.vlsthresh:
            #print('Minnow, not a Piranha!')
            return
        self.ok = True
        self.v = (wv-mean).astype(np.int16, copy=True)

    def process(self):
        cfg = self.cfg
        c,s = getCentroid(self.v[cfg.winstart:cfg.winstop], pct=0.8)

        self.vsize = len(self.v)
        self.vc = np.float16(c)
        self.vs = np.uint64(s)

def save_spec(data: List[SpectData]) -> Dict[str,Any]:
    if len(data) == 0:
        return {}
    return dict(
        config = data[0].cfg,
        events = np.array([x.event for x in data], dtype=np.uint32),
        centroids = np.array([x.vc for x in data], dtype=np.float16),
        vsum = np.array([x.vs for x in data], dtype=np.uint64),
        vsize = np.array([x.vsize for x in data], dtype=np.int32),
        # capture any raw x.v?
    )
