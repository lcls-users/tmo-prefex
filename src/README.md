# using timestamps
```bash
# from Mona for timestamps
timestamps = np.array([4194783241933859761,4194783249723600225,4194783254218190609,4194783258712780993], dtype=np.uint64)
ds = DataSource(exp='tmoc00118', run=222, dir='/sdf/data/lcls/ds/prj/public01/xtc',timestamps=timestamps)
myrun = next(ds.runs())
opal = myrun.Detector('tmo_atmopal')
print(nevt, evt.timestamp, img.shape)
ts:np.uint64 = run.timestamp
```
