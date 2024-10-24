from typing import List, Dict, Any, Union
from collections.abc import Callable

from new_port import PortConfig
from Gmd import GmdConfig

import numpy as np
from stream import filter

def load_data(ret, data) -> Dict[str,np.ndarray]:
    for name in data.keys():
        ret[name] = data[name][:]
    return ret

def concat(arg):
    if isinstance(arg, list):
        vals = arg
    else:
        vals = list(arg)
    if len(vals) == 0:
        return np.array([], dtype=np.int32)
    return np.hstack(vals)

class Batch(dict):
    """ Class to hold and concatenate a dict-of-dict-of (h5-like dicts)
        as produced by save_batch.
    """
    dtypes = {'events': np.uint32,
              # from HSD detectors
              'addresses': np.uint64,
              'nedges': np.uint64,
              'tofs': np.uint64,
              'slopes': np.int64,
              'rl_events': np.uint32,
              'rl_addresses': np.uint64,
              'raw_lens': np.uint16,
              'logic_lens': np.uint16,
              'rl_data': np.int32,
              # from GMD detectors
              'energies': np.int16,
             }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extend(self, *data):
        raise NotImplementedError("keys, datasets, tbdh")
        for k1, v1 in self.items():
            for k2, v2 in v1.items():
                ret = {}
                for name, dtype in self.datasets:
                    u = [v2[name]] + [d[k1][k2][name] for d in data]
                    if name == "addresses":
                        print("FIXME")
                        exit(1)
                        off = np.cumsum([0] + [v2[name][-1]] + [d[k1][k2]["nedges"][-1] for d in data[:-1]])
                        for x, o in zip(u, off):
                            x += o
                    if name == "rl_addresses":
                        off = np.cumsum([0] + [v2[name][-1]] + [d[k1][k2]["raw_lens"][-1]+d[k1][k2]["logic_lens"][-1] for d in data[:-1]])
                        for x, o in zip(u, off):
                            x += o
                    ret[name] = concat(u)
                v1[k2] = ret

    @classmethod
    def from_h5(cls, f):
        data = cls()

        for k, v in f.items():
            # TODO: parse k as (name, int)
            #
            # TODO: determine type of detector (from k?)
            #       and gather config. options here.
            ret = {"PortConfig": PortConfig.model_validate_json(
                                        v.attrs["PortConfig"])
                  }
            data[k] = load_data(ret, v)

        return cls(data)

    def write_h5(self, f):
        for hsdname, data in self.items():
            if hsdname in f.keys():
                nmgrp = f[hsdname]
            else:
                nmgrp = f.create_group(hsdname)

            for name, data in self.items():
                dtype = self.dataset[name]
                g.create_dataset(name,
                                 data=np.array(data, dtype=dtype),
                                 dtype=dtype)

            # FIXME: generalize to other config types.
            # Store PortConfig in a json string.
            g.attrs.create("PortConfig",
                           data=data["PortConfig"].model_dump_json())

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

def map_dict(u: List[List[Dict]],
             *fn: Callable[ [List[Any]], Dict ]
            ) -> Dict:
    """ Batch together all values from the inputs
        under their respective dictionary key.

        Transpose the indices so that the outer list
        dimension (events) is the inner one.
        
        Then, for each detector, run that detector's
        data batching function on each list.
        The result is a dict holding the function result
        as each value.
    """
    if len(u) == 0:
        return {}

    m = len(u[0]) # detectors
    n = len(u)    # events
    def mk_empty():
        return [ [None]*n for i in range(m) ]

    val = { k:mk_empty() for for k in u[0][0].keys() }

    for i,x in enumerate(u): # list elems (inner)
        for j, y in enumerate(x): # tuple elems (outer)
            for k, v in y.items():
                val[k][j][i] = v2

    ans = {}
    for k, v in val.items():
        ans[k] = fn(v)
    return ans

def save_dict_batch(elems, *fns) -> Batch:
    """ Convert a list of (hsd, gmd, ...) data tuples to a Batch.
    fns should correspond to the type of tuples passed.

    So (hsd, gmd) should use save_dict_batch(elems, save_hsd, save_gmd)
    """
    return batch_dict(elems, *fns)
