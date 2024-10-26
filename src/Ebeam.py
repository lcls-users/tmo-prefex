from typing import List, Any, Optional, Dict
from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel
from stream import stream

class EbeamConfig(BaseModel):
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
        pass

@stream
def run_ebeams(events, ebeams, params) -> Iterator[Optional[Any]]:
    for eventnum, evt in events:
        completeEvent = True

        out = {}
        for name, detector in ebeams.items():
            idx = (name, 0)
            out[idx] = EbeamData(params[idx],
                                 eventnum,
                                 detector.raw.the_ebeam_value(evt))
            completeEvent = out[idx].ok
            if not completeEvent:
                break
    if completeEvent:
        yield out
    else:
        yield None

def save_ebeam(data: List[GmdData]) -> Dict[str,Any]:
    if len(data) == 0:
        return {}
    return dict(
        config = data[0].cfg,
        events   = np.array([x.event for x in data], dtype=np.uint32),
        l3energy = np.array([x.l3 for x in data], dtype=np.float16)
    )
