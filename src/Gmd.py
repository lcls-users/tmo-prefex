from typing import Optional, Dict, Any, List
from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel

from stream import stream

class GmdConfig(BaseModel):
    name: str = 'gmd'
    unit: str = 'uJ'
    scale: int = 1000

def setup_gmds(run, params) -> Dict[str,Any]:
    # gmd-s store their output in xray-s
    gmdnames = [s for s in run.detnames if s.endswith("gmd")]

    gmds = {}
    for gmdname in gmdnames:
        gmd = run.Detector(gmdname)
        if gmd is None:
            print(f'run.Detector({gmdname}) is None!')
            continue
        gmds[gmdname] = gmd

        idx = (gmdname, 0)
        if idx not in params:
            cfg = GmdConfig(
                name = gmdname
            )
            if 'x' in gmdname: # or gmdname.endswith('x')
                cfg.unit = '0.1uJ'
                cfg.scale = 10000
            params[idx] = cfg

    return gmds

class GmdData:
    def __init__(self,
                 cfg: GmdConfig,
                 event:int,
                 energy: Optional[float]
                ) -> None:
        self.event = event
        self.cfg = cfg
        if energy is None or energy < 0:
            self.ok = False
            self.energy = 0
        else:
            self.ok = True
            self.energy = np.uint16(energy*self.cfg.scale)

@stream
def run_gmds(events, gmds, params) -> Iterator[Optional[Any]]:
    # assume rungmd is True if this fn. is called

    for eventnum, evt in events:
        completeEvent = True

        out = {}
        for gmdname, gmd in gmds.items():
            idx = (gmdname, 0)
            out[idx] = GmdData(params[idx],
                               eventnum,
                               gmd.raw.milliJoulesPerPulse(evt))
            completeEvent = out[idx].ok
            if not completeEvent:
                break

        if completeEvent:
            yield out
        else:
            yield None

def save_gmd(data: List[GmdData]) -> Dict[str,Any]:
    """ Save a batch of data.

    Gathers up relevant information from the processed data
    of a gmd-type detector.
    """
    if len(data) == 0:
        return {}
    return dict(
        config   = data[0].cfg,
        events   = np.array([x.event for x in data], dtype=np.uint32),
        energies = np.array([x.energy for x in data], dtype=np.uint16),
    )
