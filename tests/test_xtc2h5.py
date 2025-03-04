import pytest

try:
    from tmo_prefex.cmd.xtc2h5 import main
    is_loaded = True
except ModuleNotFoundError:
    main = lambda *x: None
    is_loaded = False

@pytest.mark.skipif(not is_loaded, reason="psana unavailable")
def test_main():
    with pytest.raises(KeyError):
        x = main('expt123', 7, 'mrco,polo')
