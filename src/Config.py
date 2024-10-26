#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from typing import Annotated, Any, Union, List
from pathlib import Path

import h5py
import numpy as np
import yaml
from pydantic import BaseModel, Discriminator, Tag

from Hsd import HsdConfig
from Gmd import GmdConfig

def get_detector_type(v: Any) -> str:
    if isinstance(v, dict):
        name = v.get('name')
    else:
        name = getattr(v, 'name')
    if name.endswith('hsd'):
        return 'hsd'
    elif name.endswith('gmd'):
        return 'gmd'
    #raise ValueError(f"Cannot determine type for detector: {name}")
    return None

class Config(BaseModel):
    """ A global config can be made from a list of
    per-detector config. options.
    """
    detectors: List[ Annotated[
        Union[
            Annotated[HsdConfig, Tag('hsd')],
            Annotated[GmdConfig, Tag('gmd')],
        ],
        Discriminator(get_detector_type),
    ] ] = []
    # t0s
    # logicthresh
    # offsets

    def to_dict(cfg):
        return {(d.name,getattr(d,'id',0)): d for d in cfg.detectors}

    @classmethod
    def from_dict(cls, ans):
        return cls.model_validate({'detectors':list(ans.values())})

    #expand: int # expand controls the fractional resolution for scanedges by scaling index values and then zero crossing round to intermediate integers.
    #inflate: int # inflate pads the DCT(FFT) with zeros, artificially over sampling the waveform
    #vlsthresh
    #vlswin
    #l3offset

    @classmethod
    def load(cls, fname: str):
        with open(fname, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.model_validate(cfg)

    def save(cfg, fname: str, overwrite=False) -> None:
        if not overwrite and Path(fname).exists():
            raise FileExistsError(f"won't overwrite {fname}")
        with open(fname, "w", encoding="utf-8") as f:
            yaml.dump(cfg.model_dump(), f, indent=2)

example_config = """
- name: mrco_hsd
  id: 0
  chankey: 0
  expand: 4
  inflate: 2
  is_fex: true
  logicthresh: -1*(1<<13) # set by 1st knee (log-log) in val histogram
  # offsets: [0,0,0,0]

- name: gmd
  vlsthresh: 1000
  vlswin: (1024,2048)
  l3offset: 5100
"""
''' ???
params.update({'hsdchannels':{'mrco_hsd_0':'hsd_1B_A',
            'mrco_hsd_22':'hsd_1B_B',
            'mrco_hsd_45':'hsd_1A_A',
            'mrco_hsd_67':'hsd_1A_B',
            'mrco_hsd_90':'hsd_3E_A',
            'mrco_hsd_112':'hsd_3E_B',
            'mrco_hsd_135':'hsd_3D_A',
            'mrco_hsd_157':'hsd_89_B',
            'mrco_hsd_180':'hsd_01_A',
            'mrco_hsd_202':'hsd_01_B',
            'mrco_hsd_225':'hsd_DA_A',
            'mrco_hsd_247':'hsd_DA_B',
            'mrco_hsd_270':'hsd_B2_A',
            'mrco_hsd_292':'hsd_B2_B',
            'mrco_hsd_315':'hsd_B1_A',
            'mrco_hsd_337':'hsd_B1_B'} })
'''

example_config = """
detectors:
  - name: hsd1
    id: 0
    chankey: 0
    is_fex: true
  - name: xgmd
    scale: 1000
"""

if __name__=="__main__":
    cfg1 = yaml.safe_load(example_config)
    cfg = Config.model_validate(cfg1)
    print(yaml.dump(cfg.model_dump(), indent=2))

    d = cfg.to_dict()
    cfg2 = cfg.from_dict(d)
