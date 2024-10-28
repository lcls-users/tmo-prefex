from pathlib import Path
import sys
sys.path = [p for p in sys.path if 'HydraGNN' not in p]
sys.path.append( str(Path(__file__).absolute().parent.parent / "src" ))

import numpy as np

from Ebeam import EbeamConfig, EbeamData, setup_ebeams, run_ebeams, save_ebeam

def test_config():
    x = EbeamConfig(name='zeebeam', l3offset=1234)
    EbeamConfig(name='zeebeam')

def test_data():
    cfg = EbeamConfig(name='ebeam')
    data = EbeamData(cfg, 10, None)
    assert not data.ok

    data = EbeamData(cfg, 11, 'hunh?')
    assert not data.ok

    data = EbeamData(cfg, 12, 31337)
    assert data.ok

class FakeEbeam:
    def __init__(self, en):
        self.en = en
        self.raw = self
    def ebeamL3Energy(self, evt):
        return self.en

def test_run():
    params = {('ebeam',0): EbeamConfig(name='ebeam')}
    ebeams = {'ebeam': FakeEbeam(2501)}
    vals = enumerate(range(10)) >> run_ebeams(ebeams,params) >> list
    assert len(vals) == 10

def test_save():
    cfg = EbeamConfig(name='ebeam', l3offset=5100)
    g1 = EbeamData(cfg, 10, 5400).process()
    g2 = EbeamData(cfg, 11, 10100).process()
    ans = save_ebeam([g1, g2])
    assert ans['config'] == cfg
    assert np.allclose(ans['events'], np.array([10,11]))
    assert np.allclose(ans['l3energy'], np.array([300.0, 5000.0]))
