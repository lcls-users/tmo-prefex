from pathlib import Path
import sys
sys.path = [p for p in sys.path if 'HydraGNN' not in p]
sys.path.append( str(Path(__file__).absolute().parent.parent / "src" ))

import yaml

from Config import Config

example_config = """
detectors:
  - name: mrco_hsd
    id: 0
    chankey: 0
    is_fex: true
  - name: xgmd
    scale: 1000

vlsthresh: 1000
vlswin: [1024,2048]
l3offset: 5100
"""

def test_config_dict():
    cfg1 = yaml.safe_load(example_config)
    cfg = Config.model_validate(cfg1)
    print(yaml.dump(cfg.model_dump(), indent=2))

    d = cfg.to_dict()
    cfg2 = cfg.from_dict(d)
    assert cfg2 == cfg

def test_config(tmpdir):
    cfg1 = yaml.safe_load(example_config)
    cfg = Config.model_validate(cfg1)

    cfg.save(tmpdir/'config.yaml')
    cfg2 = Config.load(tmpdir/'config.yaml')
    assert cfg == cfg2
