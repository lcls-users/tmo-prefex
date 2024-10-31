
from pathlib import Path

import numpy as np
import h5py

from tmo_prefex.combine import Batch

def test_batch():
    with h5py.File(here/"test1.h5") as f:
        batch = Batch.from_h5(f)
    print(batch)
    assert isinstance(batch, Batch)
    assert ('xgmd', 0) in batch
    assert ('gmd', 0) in batch
    assert ('mrco_hsd', 0) in batch
    assert len(batch[('mrco_hsd', 0)]) > 3
    for k, v in batch.items():
        assert 'events' in v

def test_combine():
    with h5py.File(here/"test1.h5") as f:
        batch1 = Batch.from_h5(f)
    with h5py.File(here/"test2.h5") as f:
        batch2 = Batch.from_h5(f)
    a = {k:len(v['events']) for k,v in batch1.items()}
    b = {k:len(v['events']) for k,v in batch2.items()}

    mkeys = [k for k in batch1.keys() if k[0] == 'mrco_hsd']
    st = {k: np.hstack([batch1[k]['addresses'],
                    len(batch1[k]['tofs'])+batch2[k]['addresses']]) \
                for k in mkeys}
    st2 = {k: np.hstack([batch1[k]['rl_addresses'],
                     len(batch1[k]['rl_data'])+batch2[k]['rl_addresses']]) \
                for k in mkeys}
    #print(batch1[('mrco_hsd',0)]['addresses'])
    #print(batch2[('mrco_hsd',0)]['addresses'])
    batch1.extend(batch2) # test extend-by-1 batches
    #print(batch1[('mrco_hsd',0)]['addresses'])
    for k, v in batch1.items():
        assert len(v['events']) == a[k]+b[k]
    for k in mkeys:
        assert np.allclose(st[k], batch1[k]['addresses'])
        assert np.allclose(st2[k], batch1[k]['rl_addresses'])
    batch1.extend(batch2, batch2) # test extend-by-2 batches
    for k, v in batch1.items():
        assert len(v['events']) == a[k]+3*b[k]

if __name__=="__main__":
    #test_batch()
    test_combine()
