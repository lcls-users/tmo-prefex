from typing import List, Dict, Any
from collections.abc import Callable

from Config import DetectorConfig
from Hsd import HsdConfig, WaveData, FexData
from Gmd import GmdConfig, GmdData
from Spect import SpectConfig, SpectData # was Vls?
from utils import concat

import numpy as np
from stream import filter

def load_data(cfg, data) -> Dict[str,np.ndarray]:
    # rather than HsdConfig, etc., just name the attr as 'config'
    #ret = {cfg.__class__.__name__: cfg}
    ret = {'config': cfg}
    for name in data.keys():
        ret[name] = data[name][()]
    return ret

# FIXME: should we add a runstr as top-level hdf5 group?
def get_runstr(self):
    return 'run_%04i'%self.runkey

class Batch(dict):
    """ Class to hold and concatenate a dict-of-dict-of (h5-like dicts)
        as produced by save_batch.
    """
    dtypes = {'events': np.uint32,
              'config': 'json',
              # from HSD detectors
              'addresses': np.uint64,
              'nedges': np.uint64,
              'tofs': np.uint64,
              'slopes': np.int64,
              'rl_events': np.uint32,
              'rl_addresses': np.uint64,
              'raw_lens': np.uint16,
              'logic_lens': np.uint16,
              'rl_data': np.int32, # rwl were u16,s16,s32
              # from Gmd detectors
              'energies': np.int16,
              # from Spect detectors
              #'v': int, 
              'centroids': np.float16,
              'vsum': np.uint64,
              'vsize': np.int32,
              # from Ebeam detectors
              'l3energy': np.float16,
              'l3offset': np.uint16,
             }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extend(self, *data) -> "Batch":
        # address keys and their corresponding data targets
        # - these are used to update addressing info.
        #   for appended data blocks.
        addr_keys = {
            # listing both tofs and slopes here triggers a check
            # that len(slopes) == len(tofs)
            'addresses':    ['tofs', 'slopes'],
            'rl_addresses': ['rl_data'],
        }
        data = [d for d in data if len(d) > 0]
        if len(data) == 0:
            return self
        for idx, det_data in self.items():
            assert all([idx in d for d in data]), f"Not all Batch-es contain {idx}"

            # First handle all address keys:
            for k, v in det_data.items():
                if k not in addr_keys:
                    continue
                sz_key = addr_keys[k][0]
                sz = [len(det_data[sz_key])] \
                      + [len(d[idx][sz_key]) for d in data]
                # Check that additional keys have same lengths.
                for sz_key2 in addr_keys[k][1:]:
                    sz2 = [len(det_data[sz_key2])] \
                          + [len(d[idx][sz_key2]) for d in data]
                    if tuple(sz) != tuple(sz2):
                        raise ValueError(f"Length mismatch between keys {sz_key} and {sz_key2}: {sz} vs. {sz2}")

                off = np.cumsum([0] + sz[:-1])
                u = [v] + [d[idx][k] for d in data]
                for x, o in zip(u, off):
                    x += o
                det_data[k] = concat(u)

            # Next, handle non-address data fields
            for k, v in det_data.items():
                if k in addr_keys:
                    continue
                if self.dtypes[k] == 'json':
                    continue
                det_data[k] = concat([v]+[d[idx][k] for d in data])

        return self

    @classmethod
    def concat(cls, x):
        # FIXME: remove extend in favor of concat...
        return x[0].extend(*x[1:])

    @classmethod
    def from_h5(cls, f):
        self = cls()

        for name, det in f.items():
            for chan, data in det.items():
                idx = (name, int(chan))

                # To validate the config,
                #ret = {"HsdConfig": HsdConfig.model_validate_json(
                #                         data.attrs["HsdConfig"])
                #      }
                cfgs = [v for k,v in data.attrs.items() \
                          if k.lower().endswith("config")]
                assert len(cfgs) == 1, f"{idx} does not contain a unique config in: {list(data.attrs.keys())}"
                cfg = DetectorConfig.model_validate_json(
                        '{"detector":%s}'%cfgs[0])
                self[idx] = load_data(cfg, data)

        return self

    def write_h5(self, f):
        for (name,chan), data in self.items():
            if name in f.keys():
                nmgrp = f[name]
            else:
                nmgrp = f.create_group(name)
            c = str(chan) # hdf5 won't accept numbers as keys
            if c in nmgrp.keys():
                g = nmgrp[c]
            else:
                g = nmgrp.create_group(c)

            for k, v in data.items():
                dtype = self.dtypes[k]
                if dtype == 'json':
                    # Store HsdConfig, etc. in json strings.
                    g.attrs.create(k, data=v.model_dump_json())
                else:

                    g.create_dataset(k,
                                     data=np.array(v, dtype=dtype),
                                     dtype=dtype)

def batch_data(u: List[ List[Dict] ],
               fns: List[ Callable[ [List[Any]], Dict[str,Any]] ]
              ) -> Batch:
    """ Converts a list of (hsd, gmd, ...) data tuples to a Batch.
        fns should correspond to the type of tuples passed.

        So (hsd, gmd) should use bach_data(elems, [save_hsd,save_gmd])

        Basically transposes the indices so that the outer list
        dimension (events) is the inner one.
        
        Then, for each detector, run that detector's
        data batching function on each list.
        The result is a dict holding the function result
        as each value.
    """
    if len(u) == 0:
        return Batch()
    assert len(u[0]) == len(fns), "I must have one combination function for each detector type."

    n = len(u)    # events
    def mk_empty():
        return [None]*n

    val = {}
    for det_outputs in u[0]: # First output for ea. detector type
        val.update({ k:mk_empty() for k in det_outputs.keys() })

    for i,x in enumerate(u): # list elems (inner)
        for _, y in enumerate(x): # outputs for ea. detector type
            for k, v in y.items():
                val[k][i] = v

    ans = {}
    for fn, det_outputs in zip(fns,u[0]):
        for k in det_outputs.keys():
            ans[k] = fn( val[k] )
    return Batch(ans)
