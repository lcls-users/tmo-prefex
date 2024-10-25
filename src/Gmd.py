from typing import Optional, Dict, Any, List

import numpy as np
from pydantic import BaseModel

class GmdConfig(BaseModel):
    name: str = 'gmd'
    unit: str = 'uJ'
    scale: int = 1000

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

def save_gmd(data: List[GmdData]) -> Dict[str,Any]:
    """ Save a batch of data.

    Gathers up relevant information from the processed data
    of a gmd-type detector.
    """
    if len(data) == 0:
        return {}
    return dict(
        events   = np.array([x.event for x in data], dtype=np.uint32),
        energies = np.array([x.energy for x in data], dtype=np.uint16),
        GmdConfig = data[0].cfg,
    )
