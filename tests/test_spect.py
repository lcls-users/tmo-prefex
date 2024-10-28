from pathlib import Path
import sys
sys.path = [p for p in sys.path if 'HydraGNN' not in p]
sys.path.append( str(Path(__file__).absolute().parent.parent / "src" ))

import pytest
from pydantic import ValidationError
import numpy as np

from Spect import SpectConfig, SpectData, setup_spects, run_spects, save_spect

def test_config():
    x = SpectConfig(name='z_piranha', vlsthresh=1000)

test_wv = 8000*np.exp(-0.5*(np.arange(3001)*2/3000.0 - 1.0)**2)

def test_data():
    cfg  = SpectConfig(name='abc_spect', vlsthresh=500,
                       winstart=1024, winstop=2048)
    data = SpectData(cfg, 101, test_wv)
    assert data.ok

    data = SpectData(cfg, 102, np.zeros(3000)+cfg.vlsthresh*2)
    assert not data.ok

    with pytest.raises(ValidationError):
        SpectConfig(name='z_piranha')

class FakeSpect:
    def __init__(self):
        self.raw = self
    def raw(self, evt):
        return test_wv

def test_run():
    params = {('z_piranha',0): SpectConfig(name='z_piranha',
                                           vlsthresh=100)}
    spects = {'z_piranha': FakeSpect()}
    vals = enumerate(range(10)) >> run_spects(spects,params) >> list
    assert len(vals) == 10

def test_save():
    cfg = SpectConfig(name='x_piranha', vlsthresh=1000)
    g1 = SpectData(cfg, 10, test_wv).process()
    g2 = SpectData(cfg, 11, test_wv).process()
    ans = save_spect([g1, g2])
    assert ans['config'] == cfg
    assert np.allclose(ans['events'], np.array([10,11]))
    print(ans['centroids'])
    print(ans['vsum'])
    print(ans['vsize'])
    # [1600. 1600.]
    # [880628 880628]
    # [3001 3001]

    #assert np.allclose(ans['energies'], np.array([112, 101]))
