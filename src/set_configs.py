#!/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps-4.6.3/bin/python3
import h5py
import numpy as np
import sys

def main():

    print('this is broken, move this to a yaml file for configs.')
    return

    for k in logicthresh.keys():
        logicthresh[k] = logicthresh[k]>>2

    vlsthresh = 1000
    vlswin = (1024,2048)
    l3offset = 5100
    t0s = {0:4577,1:4186,2:4323,4:4050,5:4128,12:4107,13:4111,14:4180,15:4457,16:4085} # these are not accounting for the expand nor inflate, digitizer units, 6GSps, so 6k = 1usec

    if len(sys.argv)<2:
        print('I need an output filename to write configs to')
        return

    cfgfile = sys.argv[1]
    with h5py.File(cfgfile,'w') as f:
        f.attrs.create('expand',4) # expand controls the fractional resolution for scanedges by scaling index values and then zero crossing round to intermediate integers.
        f.attrs.create('inflate',2) # inflate pads the DCT(FFT) with zeros, artificially over sampling the waveform
        f.attrs.create('vlsthresh',data=vlsthresh)
        f.attrs.create('vlswin',data=vlswin)
        f.attrs.create('l3offset',data=l3offset)
        for k in chans.keys():
            key = 'port_%i'%int(k)
            c = f.create_group(key)
            c.attrs.create('hsd',data=chans[k])
            c.attrs.create('t0',data=t0s[k])
            c.attrs.create('logicthresh',data=logicthresh[k])
            c.attrs.create('offsets',data=offsets[k])
    return

if __name__ == '__main__':
    main()

