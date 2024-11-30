# Data pre-analysis

The data pre-analysis tool, `xtc2h5`, processes
data as described in [*datasets*](docs/datasets.md).
Installing this package (e.g. with `pip install -e .`)
provides this as a command-line tool.

```
% xtc2h5 --help
                                                                         
 Usage: xtc2h5 [OPTIONS] EXPNAME DETECTORS RUN                           
                                                                         
╭─ Arguments ───────────────────────────────────────────────────────────╮
│ *    expname        TEXT     Experiment name [default: None]          │
│                              [required]                               │
│ *    detectors      TEXT     Comma-separated list of detectors (e.g.  │
│                              gmd,hsd,spect)                           │
│                              [default: None]                          │
│                              [required]                               │
│ *    run            INTEGER  Run number [default: None] [required]    │
╰───────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────╮
│ --config                    PATH  Detector configuration file         │
│                                   [default: None]                     │
│ --dial                      TEXT  Detector configuration file         │
│                                   [default: None]                     │
│ --help                            Show this message and exit.         │
╰───────────────────────────────────────────────────────────────────────╯
╭─ Output prefix for hdf5 files ────────────────────────────────────────╮
│ --outdir        PATH  Defaults to                                     │
│                       /sdf/scratch/lcls/ds/{abbr}/{expname}/scratch/… │
│                       [default: None]                                 │
╰───────────────────────────────────────────────────────────────────────╯
```

It can be launched in serial, but should normally
be launched in parallel with mpirun.
See `slurmscript.bash` for a complete example.

It can be launched with, e.g.

    ssh psana sbatch $PWD/slurmscript.bash tmox1016823 45

