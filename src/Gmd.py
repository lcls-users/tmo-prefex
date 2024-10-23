from typing import Optional

import numpy as np
from pydantic import BaseModel

class GmdConfig(BaseModel):
    unit: str = 'uJ'
    scale: float = 1000
    name: str = 'gmd'

class GmdData:
    def __init__(self, cfg: GmdConfig, energy: Optional[float]):
        self.cfg = cfg
        if energy is None or energy < 0:
            self.ok = False
            self.energy = 0
        else:
            self.ok = True
            self.energy = np.uint16(energy*1000)

class GmdRun:
    datasets = [
                ('events', np.uint32),
                ('energy', np.uint16),
            ]

class Gmd:
    def __init__(self):
        self.en = []
        self.initState = True
        return 

    @classmethod
    def slim_update_h5(cls,f,gmd,gmdEvents):
        grpgmd = None
        if 'gmd' in f.keys():
            grpgmd = f['gmd']
        else:
            grpgmd = f.create_group('gmd')
        grpgmd.create_dataset('gmdenergy',data=gmd.en,dtype=np.uint16)
        grpgmd.create_dataset('events',data=gmdEvents)
        return

    @classmethod
    def update_h5(cls,f,gmd,gmdEvents):
        grpgmd = None
        if 'gmd' in f.keys():
            grpgmd = f['gmd']
        else:
            grpgmd = f.create_group('gmd')
        grpgmd.create_dataset('gmdenergy',data=gmd.en,dtype=np.uint16)
        grpgmd.create_dataset('events',data=gmdEvents)
        return

    def test(self,e):
        if type(e)==type(None):
            return False
        if e<0:
            return False
        return True

    def process(self,e):
        if self.initState:
            self.en = [np.uint16(e*1000)]
        else:
            self.en += [np.uint16(e*1000)]
        return True

    def set_initState(self,state):
        self.initState = state
        return self
