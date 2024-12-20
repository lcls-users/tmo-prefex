from pathlib import Path

import numpy as np

from tmo_prefex.Gmd import GmdConfig, GmdData, setup_gmds, run_gmds, save_gmd

def test_config():
    x = GmdConfig(name='xgmd', unit='0.1uJ', scale=1000)

def test_data():
    cfg = GmdConfig(name='gmd')
    data = GmdData(cfg, 10, 0.112)
    assert data.ok

    data = GmdData(cfg, 10, None)
    assert not data.ok

class FakeGmd:
    def __init__(self, en):
        self.en = en
        self.raw = self
    def milliJoulesPerPulse(self, evt):
        return self.en

def test_run():
    params = {('gmd',0): GmdConfig(name='gmd', scale=1000)}
    gmds = {'gmd': FakeGmd(0.1)}
    vals = enumerate(range(10)) >> run_gmds(gmds,params) >> list
    assert len(vals) == 10

def test_save():
    cfg = GmdConfig(name='gmd', scale=1000)
    g1 = GmdData(cfg, 10, 0.112).process()
    g2 = GmdData(cfg, 11, 0.101).process()
    ans = save_gmd([g1, g2])
    assert ans['config'] == cfg
    assert np.allclose(ans['events'], np.array([10,11]))
    assert np.allclose(ans['energies'], np.array([112, 101]))
