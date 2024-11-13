# Plugin system for detector types.
#
# Each detector registers its config and data classes,
# and provides a "setup" and "run" method.
# Note these names should 

from .Hsd import HsdConfig, WaveData, FexData, run_hsds, setup_hsds, save_hsd
from .Ebeam import EbeamConfig, EbeamData, setup_ebeams, run_ebeams, save_ebeam
from .Gmd import GmdConfig, GmdData, setup_gmds, run_gmds, save_gmd
from .Spect import SpectConfig, SpectData, setup_spects, run_spects, save_spect

# Some types:
DetectorID   = Tuple[str, int] # ('hsd', 22)
DetectorData = Union[WaveData, FexData, GmdData, EbeamData, SpectData]
EventData    = Dict[DetectorID, DetectorData]

def save_fex(run, params):
    return setup_hsds(run, params, default_fex)

# To add:
# ['lcams', 'timing', 'xtcav']
# note: renamed vls <-> spect
detector_configs = {
    # NOTE: defaults to fex-type hsd setup
    'hsd': (setup_hsds, run_hsds, save_hsd),
    'ebeam': (setup_ebeams, run_ebeams, save_ebeam),
    'gmd': (setup_gmds, run_gmds, save_gmd),
    'spect': (setup_spects, run_spects, save_spect),
}
