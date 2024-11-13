from pathlib import Path

import yaml

from tmo_prefex.Config import Config

example_config = """
detectors:
  - name: mrco_hsd
    id: 0
    chankey: 0
    is_fex: true
  - name: xgmd
    scale: 1000
  - name: tmo_fzppiranha
    vlsthresh: 1000
    winstart: 1024
    winstop: 2048
"""

example_config2 = """
detectors:
  - name: tmo_fzppiranha
    vlsthresh: 1000
    winstart: 1024
    winstop: 2048
  - name: mrco_hsd
    id: 0
    chankey: 0
    is_fex: true
  - name: xgmd
    scale: 1000
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

def test_hash():
    cfg1 = yaml.safe_load(example_config)
    cfg = Config.model_validate(cfg1)
    s = cfg.hash()
    print(s)

    cfg2 = yaml.safe_load(example_config2)
    s2 = Config.model_validate(cfg2).hash()
    assert s == s2
