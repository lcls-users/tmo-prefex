from typing import List, Any, Optional, Dict
from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel
from stream import stream

class EbeamConfig(BaseModel):
    name: str
    l3offset: float = 5100 # int

class EbeamData:
    def __init__(self,
                 cfg: EbeamConfig,
                 event: int,
                 l3in):
        self.cfg = cfg
        self.event = event

        self.ok = False
        if l3in is None:
            return

        try:
            self.l3 = np.float16(float(l3in)-float(cfg.l3offset))
        except:
            print('Damnit, Ebeam!')
            return
        self.ok = True

    def process(self):
        return self

def setup_ebeams(run, params):
    enames = [s for s in run.detnames if s.endswith("ebeam")]

    ebeams = {}
    for name in enames:
        beam = run.Detector(name)
        if beam is None:
            print(f'run.Detector({name}) is None!')
            continue
        ebeams[name] = beam

        idx = (name, 0)
        if idx not in params:
            cfg = EbeamConfig(
                name = gmdname,
                l3offset = 5100
            )
            params[idx] = cfg
    return ebeams

@stream
def run_ebeams(events, ebeams, params) -> Iterator[Optional[Any]]:
    for eventnum, evt in events:
        completeEvent = True

        out = {}
        for name, detector in ebeams.items():
            idx = (name, 0)
            out[idx] = EbeamData(params[idx],
                                 eventnum,
                                 detector.raw.ebeamL3Energy(evt))
            completeEvent = out[idx].ok
            if not completeEvent:
                break

        if completeEvent:
            yield out
        else:
            yield None

def save_ebeam(data: List[EbeamData]) -> Dict[str,Any]:
    if len(data) == 0:
        return {}
    return dict(
        config = data[0].cfg,
        events   = np.array([x.event for x in data], dtype=np.uint32),
        l3energy = np.array([x.l3 for x in data], dtype=np.float16)
    )
