# Testing out mpi capabilities of psana.DataSource
#
import psana

comm = psana.MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()
print(f"{rank} {ranks}")

# salloc -n4 -c1 --partition roma
# mpirun python mpidata.py

ds = psana.DataSource(exp='tmox1016823', run=215)
print(ds.is_mpi())
run = next(ds.runs())
t0 = run.timestamp
#if ranks == 1 or rank > 1: # no progress
#if ranks == 1 or rank > 0: # no progress
#if ranks == 1 or rank != 1: # no progress
if True:
    for i, evt in enumerate(run.events()):
        t = evt.timestamp_diff(t0)
        print(f"{rank} {t}")
        if i >= 100:
            break
    # we could create a sub-communicator for ranks 2+
    # and then use an MPI_Barrier on that to synchronize termination.

"""
only ranks 2, 3, ..., report
time-deltas include: 120616, 120615, 108071385, 30154, 90462
"""
