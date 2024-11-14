Datasets
========

xtc2h5 creates HDF5-formatted datasets by looping through
runs, steps, and events read by psana in parallel.
Parallel event processing is typically very fast, but can scatter
the events from each run/step into multiple files.

This is actually a good thing when correlating events --
since data from a given event is contained in only one file.
Because of this, we can parallelize data analysis over events.
Each worker sees all data pertaining to a batch of events,
but gets to work batch-at-a-time.

This document describes the data layout used by xtc2h5
to make this possible.  In the future,
[lclstream](https://github.com/lcls-users/lclstream)
will include a utility, `push_h5`, to send these files
to a client program for analysis.  For now, just copy them
over via scp.

The originating data path is typically

    /sdf/data/lcls/ds/tmo/$expname/xtc/$expname-rNNNN-s001-c000.xtc2

and processed output files are generally stored on the
scratch filesystem at

    /sdf/scratch/lcls/ds/tmo/$expname/scratch/xtc2h5/run_NNN/<config-hash>/step_MM[-rank].JJJ.h5

The `<config-hash>` value is an 8-character value determined from
a hash of the detector config file.
The `step_MM` suffix names the step as defined within the experiment
(e.g. a sweep over incoming beam frequencies). The optional `-rank` suffix gives the MPI rank writing that particular file.
The last number, `JJJ`, contains a subset of events processed
by that rank.  The rank and `J` indices could technically be
combined, since each rank and `J` contains unique events.

## File Layout

Within each file, there is a top-level group naming the step.
Inside that group, there are sub-groups for each detector.
Within each detector, there is a sub-group for a channel.
Most detectors (e.g. xgmd) have just one output channel,
but the `mrco_hsd` has 16 different channels:

```
$ h5dump -n 1 $fname
HDF5 "/sdf/data/lcls/ds/tmo/tmox1016823/scratch/rogersdd/h5files/hits.tmox1016823.run_045.step_10-085.h5" {
FILE_CONTENTS {
 group      /
 group      /step_10
 attribute  /step_10/hf_w
 attribute  /step_10/run
 attribute  /step_10/step_docstring
 group      /step_10/gmd
 group      /step_10/gmd/0
 attribute  /step_10/gmd/0/config
 dataset    /step_10/gmd/0/energies
 dataset    /step_10/gmd/0/events
 group      /step_10/mrco_hsd
 group      /step_10/mrco_hsd/0
 attribute  /step_10/mrco_hsd/0/config
 dataset    /step_10/mrco_hsd/0/addresses
 dataset    /step_10/mrco_hsd/0/events
 dataset    /step_10/mrco_hsd/0/logic_lens
 dataset    /step_10/mrco_hsd/0/nedges
 dataset    /step_10/mrco_hsd/0/raw_lens
 dataset    /step_10/mrco_hsd/0/rl_addresses
 dataset    /step_10/mrco_hsd/0/rl_data
 dataset    /step_10/mrco_hsd/0/rl_events
 dataset    /step_10/mrco_hsd/0/slopes
 dataset    /step_10/mrco_hsd/0/tofs
 group      /step_10/mrco_hsd/112
 attribute  /step_10/mrco_hsd/112/config
 dataset    /step_10/mrco_hsd/112/addresses
 dataset    /step_10/mrco_hsd/112/events
 dataset    /step_10/mrco_hsd/112/logic_lens
 dataset    /step_10/mrco_hsd/112/nedges
 dataset    /step_10/mrco_hsd/112/raw_lens
 dataset    /step_10/mrco_hsd/112/rl_addresses
 dataset    /step_10/mrco_hsd/112/rl_data
 dataset    /step_10/mrco_hsd/112/rl_events
 dataset    /step_10/mrco_hsd/112/slopes
 dataset    /step_10/mrco_hsd/112/tofs
 ...
 group      /step_10/tmo_fzppiranha
 group      /step_10/tmo_fzppiranha/0
 attribute  /step_10/tmo_fzppiranha/0/config
 dataset    /step_10/tmo_fzppiranha/0/centroids
 dataset    /step_10/tmo_fzppiranha/0/events
 dataset    /step_10/tmo_fzppiranha/0/offsets
 dataset    /step_10/tmo_fzppiranha/0/vsize
 dataset    /step_10/tmo_fzppiranha/0/vsum
 dataset    /step_10/tmo_fzppiranha/0/wv
 group      /step_10/xgmd
 group      /step_10/xgmd/0
 attribute  /step_10/xgmd/0/config
 dataset    /step_10/xgmd/0/energies
 dataset    /step_10/xgmd/0/events
 }
}
```

Attributes included on the step group provide the output
of the step's "detectors":
```
$ h5dump -a '/step_10/hf_w' $fname
HDF5 "/sdf/data/lcls/ds/tmo/tmox1016823/scratch/rogersdd/h5files/hits.tmox1016823.run_045.step_10-085.h5" {
ATTRIBUTE "hf_w" {
   DATATYPE  H5T_IEEE_F64LE
   DATASPACE  SCALAR
   DATA {
   (0): 410
   }
}
}
$ h5dump -a '/step_10/step_docstring' $fname
HDF5 "/sdf/data/lcls/ds/tmo/tmox1016823/scratch/rogersdd/h5files/hits.tmox1016823.run_045.step_10-085.h5" {
ATTRIBUTE "step_docstring" {
   DATATYPE  H5T_STRING {
      STRSIZE H5T_VARIABLE;
      STRPAD H5T_STR_NULLTERM;
      CSET H5T_CSET_UTF8;
      CTYPE H5T_C_S1;
   }
   DATASPACE  SCALAR
   DATA {
   (0): "{"detname": "scan", "scantype": "scan", "step": 10}"
   }
}
}
```

### XGMD

Within a detector's output, there are typically several variables
that output for each event.  These are stored compressed, listing
only events that have data to report.  The simplest example are the
energies for the xgmd detector:

```
$ h5ls -d $fname/step_10/xgmd/0/energies
energies                 Dataset {999}
    Data:
        (0) 90, 98, 76, 73, 108, 91, 83, 100, 97, 84, 96, 92, 108,
        (13) 98, 75, 102, 104, 91, 104, 69, 86, 68, 81, 73, 73, 87,
        (26) 104, 111, 41, 62, 83, 86, 57, 86, 81, 72, 68, 95, 92,
        (39) 97, 90, 97, 70, 72, 96, 87, 70, 84, 79, 78, 69, 72, 62,
        ...
$ h5ls -d $fname/step_10/xgmd/0/events
events                   Dataset {999}
    Data:
        (0) 51319795, 51319916, 51320036, 51320157, 51320277,
        (5) 51320398, 51320519, 51320639, 51320760, 51320880,
        (10) 51321001, 51321122, 51321242, 51321363, 51321484,
        (15) 51321604, 51321725, 51321845, 51321966, 51322087,
        (20) 51322207, 51322328, 51322448, 51322569, 51322690,
        ...
```

These are output for every event, and contain the events list
only as a tracking aid.  Note that the event numbering corresponds
to its time delta from the start of the run.

## MRCO\_HSD

The HSD detectors store only electron detections.
Hence, while there were 1000 events with xgmd data in the
file, only 57 of those detected any electrons.
```
$ h5ls $fname/step_10/mrco_hsd/180
addresses                Dataset {82}
events                   Dataset {82}
logic_lens               Dataset {1}
nedges                   Dataset {82}
raw_lens                 Dataset {1}
rl_addresses             Dataset {1}
rl_data                  Dataset {76}
rl_events                Dataset {1}
slopes                   Dataset {194}
tofs                     Dataset {194}
$ h5ls -d $fname/step_10/mrco_hsd/180/events
$ h5ls -d $fname/step_10/mrco_hsd/180/events
events                   Dataset {82}
    Data:
        (0) 51320398, 51321966, 51324378, 51325946, 51327756,
        (5) 51328117, 51328600, 51329082, 51329806, 51330047,
        (10) 51331012, 51331615, 51332580, 51333786, 51334269,
        (15) 51337043, 51337284, 51337887, 51341023, 51344039,
        (20) 51346089, 51349104, 51350672, 51351396, 51352964,
        (25) 51354532, 51356944, 51363216, 51363940, 51365508,
        ...
```

There were 194 counts totalled from these 82 events.
The `nedges` and `addresses` fields are the number
of counts in each event, and their running sum.
```
$ h5ls -d $fname/step_10/mrco_hsd/180/nedges
nedges                   Dataset {82}
    Data:
        (0) 3, 4, 4, 1, 1, 1, 2, 4, 3, 2, 5, 1, 2, 2, 1, 3, 2, 4, 1,
        (19) 1, 3, 3, 6, 2, 4, 2, 5, 1, 1, 2, 3, 1, 3, 3, 2, 3, 3, 2,
        (38) 3, 1, 1, 2, 1, 2, 4, 1, 3, 3, 4, 3, 2, 2, 2, 2, 1, 2, 4,
        (57) 1, 2, 4, 2, 5, 1, 3, 3, 2, 3, 1, 1, 3, 2, 1, 2, 3, 4, 1,
        (76) 3, 1, 1, 2, 2, 2
$ h5ls -d $fname/step_10/mrco_hsd/180/addresses
addresses                Dataset {82}
    Data:
        (0) 0, 3, 7, 11, 12, 13, 14, 16, 20, 23, 25, 30, 31, 33, 35,
        (15) 36, 39, 41, 45, 46, 47, 50, 53, 59, 61, 65, 67, 72, 73,
        (29) 74, 76, 79, 80, 83, 86, 88, 91, 94, 96, 99, 100, 101,
        (42) 103, 104, 106, 110, 111, 114, 117, 121, 124, 126, 128,
        (53) 130, 132, 133, 135, 139, 140, 142, 146, 148, 153, 154,
        (64) 157, 160, 162, 165, 166, 167, 170, 172, 173, 175, 178,
        (75) 182, 183, 186, 187, 188, 190, 192
```

The `addresses` and `nedges` fields can be used to select
just the counts belonging to a single event.  Chosing here
the 15th event (51337043),
```
$ h5dump -d /step_10/mrco_hsd/180/tofs -s 36 -c 3 $fname
HDF5 "/sdf/data/lcls/ds/tmo/tmox1016823/scratch/rogersdd/h5files/hits.tmox1016823.run_045.step_10-085.h5" {
DATASET "/step_10/mrco_hsd/180/tofs" {
   DATATYPE  H5T_STD_U64LE
   DATASPACE  SIMPLE { ( 194 ) / ( 194 ) }
   SUBSET {
      START ( 36 );
      STRIDE ( 1 );
      COUNT ( 3 );
      BLOCK ( 1 );
      DATA {
      (36): 6092, 6103, 6123
      }
   }
}
}
```

## Piranha

The piranha detector captures spectral data for each event.
It also attempts to report a spectral peak as its `centroid` value.
When no spectral peak is formed, it sets `centroid=0`.
This null value is important, since `events` needs to be full
(every event has a recorded spectrum).
Spectral data is packed in `wv`, with event-specific starts
and sizes stored in `offsets` and `vsize`.
Right now these are all 2048\***n**, but eventually
the recorded spectrum sizes may differ or be centered
around a peak or something.

```
$ h5ls $fname/step_10/tmo_fzppiranha/0/
centroids                Dataset {999}
events                   Dataset {999}
offsets                  Dataset {999}
vsize                    Dataset {999}
vsum                     Dataset {999}
wv                       Dataset {2045952}

$ h5ls -d $fname/step_10/tmo_fzppiranha/0/centroids
centroids                Dataset {999}
    Data:
        (0) 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        (19) 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ...

$ h5dump -d /step_10/tmo_fzppiranha/0/wv -s 1024000 -c 2048 $fname
HDF5 "/sdf/data/lcls/ds/tmo/tmox1016823/scratch/rogersdd/h5files/hits.tmox1016823.run_045.step_10-085.h5" {
DATASET "/step_10/tmo_fzppiranha/0/wv" {
   DATATYPE  H5T_STD_I16LE
   DATASPACE  SIMPLE { ( 2045952 ) / ( 2045952 ) }
   SUBSET {
      START ( 1024000 );
      STRIDE ( 1 );
      COUNT ( 2048 );
      BLOCK ( 1 );
      DATA {
      (1024000): 7, -3, -2, -5, 4, 7, 8, 7, 6, 7, -4, 1, 1, -3, 5, 5, 1, 0,
      (1024018): -4, 5, 1, 6, -3, -3, 9, -4, -5, -1, 1, -1, 7, 3, 8, 5, -4,
      (1024035): 3, 3, 7, 0, 0, 0, 9, -3, -1, -3, 2, 3, 1, -2, -6, -8, 4, 4,
      (1024053): -5, -1, 4, 0, 0, -5, -5, 6, 2, -2, -4, 0, 0, -5, 1, 4, 5, 1,
      (1024071): 8, 0, 5, -4, 9, 0, 3, 1, 5, 7, 9, -8, -1, 6, 0, 2, -3, -4,
      (1024089): 7, 2, -1, 2, -4, 0, -1, 2, 3, -7, 1, 6, 6, 7, -1, 5, -2, -2,
      (1024107): 3, 4, 6, 1, 1, 1, 1, -7, 3, -3, -7, 1, 0, 2, 2, -4, -4, 1,
      (1024125): 4, 0, 8, 2, 3, -1, 1, -3, 0, 3, 3, 8, 3, 1, -1, -3, 5, 0, 5,
      (1024144): 0, 4, -2, -2, 0, 4, 6, 4, 1, -3, -2, 6, 0, 0, 10, 3, 4, 0,
      (1024162): 0, -7, 6, 0, 6, -2, 0, 4, -5, 2, 3, -6, 2, 5, 0, -2, 0, 1,
      (1024180): 4, 0, 6, 0, -4, 1, 3, -1, 1, -1, -1, 4, 5, 5, -5, 4, 7, -2,
      (1024198): 7, 1, 3, 9, 0, -2, 2, -4, 4, 5, 2, 11, 4, 4, -5, -2, 8, 1,
      (1024216): 4, 0, -6, -2, 4, -3, 5, -3, 10, 5, 0, -1, 3, -4, 6, 3, 6, 6,
      (1024234): 1, -2, -4, -2, 2, 1, 1, 6, -13, -3, -4, 7, 6, 2, 4, 2, 0,
      (1024251): -1, 4, 2, 2, 8, -2, 1, -1, 1, -3, -2, 3, -2, 2, 4, -4, 2,
      (1024268): -4, -5, 7, 1, 8, 4, -4, 5, 0, -6, 3, 3, 5, 1, 1, 3, 3, 4,
      (1024286): -1, 3, 6, -2, -4, 3, -5, 0, -5, 3, 10, 5, -1, -2, 1, 5, -2,
      (1024303): 7, 2, 5, -7, -4, 4, -4, 0, 0, 1, 3, -5, 3, 1, 2, 2, 4, 10,
      (1024321): 1, -7, -3, -1, 7, 1, 10, 9, 8, -3, -2, 2, 2, 2, 1, 2, 2, 2,
      (1024339): 1, 3, -1, 1, 3, 5, 3, -6, 2, 3, 2, 6, 5, 3, 6, 1, -4, 11, 2,
      (1024358): 5, -3, 5, 8, 0, 0, 3, -3, -6, 4, 0, 0, -4, -1, 4, -3, 9, 4,
      (1024376): 2, 6, -1, -3, 4, 3, 6, 8, 0, 2, 2, 1, 2, 3, 1, 6, 4, -2, 7,
      (1024395): 0, 2, -3, -1, -1, 1, 2, -9, -4, 1, -7, 1, 4, 3, 3, -5, -2,
      (1024412): 3, 1, 13, 9, 5, 4, -7, 1, 7, 4, 11, 2, 2, -1, -3, 1, 2, 2,
      (1024430): 2, 4, 0, -2, -3, 1, 2, 3, 5, 2, 2, 8, 2, -3, 2, 0, -1, 2, 9,
      (1024449): 1, 3, -1, 3, 4, 2, 2, 4, 4, -2, -1, 4, 4, -1, -5, -3, -2,
      (1024466): -3, 5, 0, 0, 3, 4, 1, -2, -1, 2, 3, -1, 4, 0, 5, 0, -7, 3,
      (1024484): 3, -1, -1, 4, -4, 1, -5, 3, 1, -2, 4, -1, 3, 3, -3, 0, -2,
      (1024501): 5, 6, 3, 3, 6, -3, -2, 3, 1, 3, 3, 0, -1, -2, 1, 2, -1, 8,
      (1024519): 2, 3, 2, 1, 3, 6, 1, 2, 2, 3, 5, -3, 3, 5, -4, 9, -1, 0, 8,
      (1024538): 6, 1, -5, 3, 3, 2, 6, 5, -2, 7, -4, 2, 6, 3, 3, -3, 0, -3,
      (1024556): -1, -1, 4, 4, 6, 4, 0, -7, 3, 3, -2, -1, 5, 2, -6, 0, 2, -3,
      (1024574): 7, 5, 2, 3, -8, -3, 2, -3, 1, 1, 8, 7, 0, 1, 3, -2, 2, 2, 7,
      (1024593): 7, -3, 2, 8, -4, 3, 2, 7, 8, -2, 0, 3, -5, 3, 2, 5, 9, 1, 0,
      (1024612): -1, 2, 9, 3, 7, -6, -2, -2, 5, 1, 2, -1, 6, 6, 5, 3, -6, -2,
      (1024630): -2, 1, 8, -2, -3, 6, 6, 1, 3, 4, -1, 6, 3, 1, 1, -2, 5, -4,
      (1024648): 2, 7, -1, -5, 8, 2, 3, 1, 11, 11, 0, 2, 0, 2, -1, -2, 7, 1,
      (1024666): -5, 7, 4, -2, 4, 3, 7, 0, 2, -2, 5, -1, 6, 6, 0, 2, -1, 3,
      (1024684): 0, -6, 2, 2, -1, -3, 3, 1, 5, -4, -2, 3, 3, 7, -6, 1, 2, 1,
      (1024702): 3, 5, 0, 5, -5, -2, 5, 1, -4, 5, -3, 6, -2, -3, 1, -2, 4,
      (1024719): -2, 3, 5, -1, 2, 6, 2, 1, 0, 6, 2, -1, -7, 5, 3, 7, 3, 4, 1,
      (1024738): -2, -1, -1, -4, 0, 11, 3, 7, 8, 1, 2, 1, 5, 0, 3, 2, -8, -3,
      (1024756): 4, 2, 2, 5, 5, 5, -6, 1, -1, -3, 11, 3, 0, -3, -1, -2, 4, 4,
      (1024774): 9, 2, 3, 4, 2, -1, 4, -2, -1, 8, -3, 3, -5, 4, 4, -9, 4, 0,
      (1024792): 5, 4, -5, 4, -3, -7, 6, 4, 3, 1, -4, 1, -4, -4, 7, 1, 0, 6,
      (1024810): -5, 1, 8, 1, 9, 0, 0, -1, -4, -3, -2, -2, 1, 2, 6, 2, -2, 1,
      (1024828): -3, 2, 4, 0, 1, 3, -2, 1, 0, 1, 7, 5, 2, -5, -2, 3, 6, -1,
      (1024846): 6, 3, -2, 5, 2, 0, 7, 1, 10, 11, 1, 4, 4, 5, 0, 0, 5, 4, 4,
      (1024865): 12, -9, 0, -1, 8, 8, 5, 1, -3, -8, 3, 2, -3, 6, 4, 7, 6, -8,
      (1024883): 10, 6, -3, 6, 4, 7, 6, -2, 6, 2, 3, -1, 2, 4, 2, -1, 5, 8,
      (1024901): -5, 2, 3, 7, 9, 6, 2, 5, 5, 3, 2, 8, 2, 2, -5, -2, -8, 8, 0,
      (1024920): 6, 10, 2, 4, 5, 6, 5, 7, 12, 14, -2, -3, 5, 1, 5, 4, 8, 4,
      (1024938): 4, 3, 6, 3, 3, 3, 5, 9, -6, 2, 12, 3, 4, 4, 12, 2, 1, 0, 6,
      (1024957): 5, 9, 7, 2, 1, -1, -3, 1, 0, 5, 5, 5, 9, -2, -1, 11, 3, 5,
      (1024975): 8, 5, 11, 0, 1, 8, 2, -3, 6, 7, 7, -2, -2, 5, -1, 4, 3, 1,
      (1024993): 9, 3, -1, 0, 4, 8, -1, 7, 8, -4, 3, 3, -1, 8, 7, 1, 9, -1,
      (1025011): -1, 8, -5, 8, 0, 6, 9, 6, 1, 3, 4, 9, 4, -1, 7, -7, -1, 0,
      (1025029): 0, -6, 0, 4, 3, -5, 2, 3, -1, 2, 1, 5, 3, -1, -4, -3, 1, -3,
      (1025047): -4, 9, 10, -5, 2, 4, 1, 6, 0, -4, 8, -8, 4, -1, -6, -1, 2,
      (1025064): -3, 5, -2, 3, 2, 7, -2, -1, -4, 3, -7, 0, 1, -5, -1, 4, 5,
      (1025081): 6, -12, 5, 2, -3, 5, 2, 3, 9, -2, 1, -4, -7, -4, 2, 2, 5,
      (1025098): -5, 2, 3, 6, -4, -1, -5, 3, -3, -4, 0, -1, -9, -5, 1, 7, -6,
      (1025115): -2, 4, 1, -2, 2, 4, 3, -7, 2, 3, 1, 1, -3, 0, 6, -6, 1, -1,
      (1025133): 4, -3, 2, 2, 8, -3, 2, 0, -1, -2, 1, 2, 5, -3, 4, 7, 0, -1,
      (1025151): 4, 9, 9, -6, -1, 2, 1, 5, 3, 5, 4, -3, 3, 5, 0, -6, -3, 10,
      (1025169): 5, 5, 4, 6, 5, -3, 4, 7, 10, 4, 10, 5, 2, 3, 6, 4, 18, 8,
      (1025187): 17, 15, 9, 9, 5, 18, 26, 10, 7, 23, 23, 17, 12, 17, 35, 23,
      (1025203): 20, 27, 23, 22, 14, 29, 55, 42, 54, 67, 88, 94, 122, 112,
      (1025217): 103, 97, 101, 82, 77, 65, 86, 73, 81, 56, 62, 73, 64, 60,
      (1025231): 56, 56, 81, 63, 69, 64, 46, 51, 40, 44, 46, 31, 42, 39, 38,
      (1025246): 38, 28, 30, 39, 26, 35, 36, 28, 26, 40, 31, 39, 48, 46, 40,
      (1025261): 38, 37, 35, 38, 45, 32, 24, 21, 17, 20, 14, 22, 25, 13, 23,
      (1025276): 30, 29, 28, 22, 27, 28, 26, 29, 30, 31, 44, 35, 29, 28, 27,
      (1025291): 38, 25, 25, 21, 21, 16, 26, 7, 28, 19, 12, 17, 14, 15, 21,
      (1025306): 8, 14, 14, 13, 9, 14, 19, 12, -2, 6, 9, 0, 6, 4, 7, 20, 7,
      (1025323): 3, 6, 6, 6, 1, 1, 6, -2, 8, 6, -3, 2, 6, 4, 8, -2, 4, 5, -1,
      (1025342): 7, 5, -6, 10, -2, -3, 1, 2, 4, 5, 3, 6, 0, 4, 4, -6, -6, -1,
      (1025360): 7, 9, 2, -1, 2, 5, 0, 4, -1, 9, -2, 3, 0, 1, -6, -5, 6, 6,
      (1025378): -7, -4, 0, -1, -3, -1, 5, -3, -2, -2, 8, 0, -9, 4, 1, 1, -3,
      (1025395): -1, 5, -1, 0, -7, 6, 6, -7, 0, 2, -3, 3, 2, 0, 5, -8, -1, 4,
      (1025413): 3, -1, -3, 0, 7, -10, 1, 3, 2, -5, -1, 5, -2, -4, 1, 6, 0,
      (1025430): -7, -1, 4, 3, -4, -4, -2, 6, -2, -2, 7, 3, -5, 3, -2, 1, 1,
      (1025447): 3, -5, 3, -3, 5, 1, 5, 0, -3, 1, 4, -14, -1, 4, 1, -10, -3,
      (1025464): 0, -1, -9, 2, 3, 4, 5, 2, 1, 0, -7, 7, 4, 4, -1, 1, 1, 7,
      (1025482): -11, -4, 1, 2, -1, 6, 3, 1, -10, -1, -2, -1, -3, -1, -4, 2,
      (1025498): -7, 0, 0, -3, 0, 1, 1, 3, -3, -4, 0, -3, 0, -4, 4, 4, -3, 1,
      (1025516): 2, -5, -4, -3, -1, 2, -6, 2, 10, 1, -2, 3, 5, 3, -6, -1, 3,
      (1025533): 6, -5, -4, 3, 2, -6, -9, -1, -1, -6, -4, 2, 7, -4, -3, -3,
      (1025549): -5, -5, -1, -4, 3, -1, -2, -4, -1, 0, 1, -4, 7, -7, -2, 5,
      (1025565): -6, -2, -3, -1, 2, -5, 1, -1, -5, 0, -1, 0, 3, -14, -3, 5,
      (1025581): -3, 3, 1, 2, 5, -3, 0, 6, -2, -2, -4, 0, 5, -6, 8, 0, 4, 0,
      (1025599): 0, 4, 2, -3, -7, 8, -3, 2, 4, 1, -3, -4, 0, 6, -1, -2, -2,
      (1025616): 2, 1, -5, -9, 8, -1, 2, 0, 2, 3, -2, -3, 6, -5, -5, 5, 1,
      (1025633): -2, -9, -2, 0, 2, 1, 4, -1, 5, -4, -3, 3, 0, 2, 3, 0, 9, -6,
      (1025651): -3, 0, -1, -4, 1, 1, 3, -6, -3, 7, -3, -4, 5, -1, 5, -10,
      (1025667): -5, 2, 1, -2, 1, 3, 3, -9, -3, 4, 5, -2, -4, 3, 0, -7, -3,
      (1025684): 2, 6, -11, -3, 2, 6, -9, -1, 5, -2, -1, 2, 1, 5, -9, 3, 5,
      (1025701): 3, -4, -4, 1, 11, -5, -1, 7, 1, -1, 1, 2, 0, -10, -5, 2, -2,
      (1025718): -3, 0, -4, 6, -2, 2, 2, 5, -5, -1, 8, 3, -1, -5, -1, 0, -2,
      (1025735): 5, 0, 3, -6, 1, 8, -2, 0, -7, 3, 2, -2, 0, 5, -6, -6, 2, -1,
      (1025753): 6, -7, 0, 8, 3, -1, 1, 1, 3, -10, -2, 0, 1, -2, -1, 4, 0,
      (1025770): -5, 0, 5, 4, -2, 3, 4, 7, -5, 3, 12, 3, -3, 2, 3, 7, -1, -1,
      (1025788): 1, -1, 1, 1, 2, 1, 3, -6, 9, -1, 2, 3, -3, 8, -4, -2, 3, -4,
      (1025806): -2, 1, 10, 6, -7, -3, 8, 4, -4, -4, 2, 2, -8, -3, 8, 3, -6,
      (1025823): -4, 3, 0, -4, -10, 9, 0, 0, 0, 4, 3, -5, 0, 6, 4, -1, -2, 2,
      (1025841): 0, -7, -6, 9, 2, -1, -2, -1, 5, -5, 1, 2, 3, -5, -1, 6, 7,
      (1025858): -8, -1, 5, -3, -5, 1, -1, 6, -6, -1, 8, 8, 0, 6, -3, 4, -5,
      (1025875): -1, 5, -1, 3, -5, 4, 6, -9, -4, 2, 0, 3, -1, -1, 9, -10, -2,
      (1025892): 8, 0, -4, -5, -1, 5, -6, 0, 9, 4, -1, 1, 11, 8, -6, -6, 8,
      (1025909): 8, -4, 3, -4, 4, -9, 0, 6, 6, 6, 4, 10, 3, -7, -3, 2, 0, -2,
      (1025927): 2, 2, 4, -4, 1, 1, -4, -3, -2, 1, -3, -6, 1, 1, 2, -1, 2,
      (1025944): -2, 9, -4, -3, 2, 3, -1, -5, 0, 1, -5, -3, 4, -3, -8, -3,
      (1025960): -1, 9, -6, 0, 7, 1, -3, -1, 3, 4, -8, -3, 7, -1, -3, 1, 0,
      (1025977): 12, -9, 3, 2, 2, -1, 4, 1, 8, -13, 2, 7, 1, -6, 4, 0, -3,
      (1025994): -1, 1, -1, -5, -3, 2, 4, 6, -4, -3, 5, -1, -6, 2, 7, 8, -8,
      (1026011): 2, 3, 7, -6, -7, 8, 2, -9, -3, 5, -7, 1, -1, 3, 6, -1, -3,
      (1026028): 3, -5, -9, 6, 7, 3, -5, 5, 1, -2, 7, 5, -1, 4, -6, 0, 8, 3,
      (1026046): -2, -1
      }
   }
}
}
```
These spectra were baseline-subtracted, accounting for the negative
numbers.  Note there is some activity around `1025203-1024000=1203`.
We could modify fex2h5 to discard data from Spect, Gmd, etc.
when none of the HSD-s register a count, but that isn't implemented
at present.

