from typing import TypeVar, Generic, List, Dict, Any
T = TypeVar('T')

import numpy as np

try:
    raise ImportError
    import cupy as xp
except ImportError:
    #import numpy as xp
    xp = np

def correl(C, A, B):
    """ Compute C += A.T @ B

    requires len(A) == len(B)
    """
    C += A.T @ B

class Quantizer(Generic[T]):
    """ A quantizer takes an object of type T
    and turns it into a histogram-like output
    of shape (rows, nbins)
    """
    def __init__(self, start, stop, nbins) -> None:
        self.start = start
        self.stop = stop
        self.nbins = nbins

    def rows(self, data: T) -> int:
        return 0

    def bin(self, xvals: xp.ndarray) -> xp.ndarray:
        scale = self.nbins / (self.stop - self.start)
        bins = ( (xvals-self.start)*scale ).astype(int)
        return bins.clip(0, self.nbins-1)

    def __call__(self, data: T, A: xp.ndarray) -> None:
        # The following should be true:
        #assert A.shape == (self.rows(data), self.nbins)
        pass

class PassThroughQ(Quantizer[Any]):
    """ Pass through a sub-array of the input.
        like Event... - takes a Tuple (FIXME: the type annotation)
    """
    def __init__(self, start, stop,
                 offsets='offsets',
                 lengths='vsize',
                 vals='wv') -> None:
        super().__init__(start, stop, stop-start)
        self.offsets = offsets
        self.lengths = lengths
        self.vals = vals

    def rows(self, data: Any) -> int:
        return max(data[0])+1

    def __call__(self, data: Any, A: xp.ndarray) -> None:
        indices, d = data
        vals = d[self.vals]
        for i,m,n in zip(indices, d[self.offsets], d[self.lengths]):
            start = self.start
            end = min(n, self.stop)
            print(m, start, end)
            print(m+start, m+end)
            m = int(m)
            A[i, 0:end-start] = vals[m+start:m+end]

class PackedQuantizer(Quantizer[xp.ndarray]):
    """ Quantize "packed" data, where:

    - data['addresses'] is a list of offsets,
    - data['nedges'] is a list of lengths,
    - data['tofs'] is the index to put into the histogram bin
    """

    def __init__(self, start, stop, nbins,
                 offsets='addresses',
                 lengths='nedges',
                 vals='tofs'
                ) -> None:
        super().__init__(start, stop, nbins)
        self.offsets = offsets
        self.lengths = lengths
        self.vals = vals

    def rows(self, data) -> int:
        n = len(data[self.offsets])
        assert n == len(data[self.lengths])
        return n

    def __call__(self, data, A: xp.ndarray) -> None:
        # calculate all bins up-front
        vals = self.bin( xp.asarray( data[self.vals] ) )
        for i,(m,n) in enumerate(zip(data[self.offsets], data[self.lengths])):
            A[i] = xp.bincount(vals[m:m+n], minlength=self.nbins)

class EventPackedQuantizer(Quantizer[xp.ndarray]):
    """ Quantize "packed" data, where:

    data = (indices, d)
    - indices: list of row numbers to put event data (1 per event)
    
    - d[___] =
       * offsets: list of offsets (1 per event)
       * lengths: list of lengths (1 per event)
       * vals: the data object
    """

    def __init__(self, start, stop, nbins,
                 offsets='addresses',
                 lengths='nedges',
                 vals='tofs'
                ) -> None:
        super().__init__(start, stop, nbins)
        self.offsets = offsets
        self.lengths = lengths
        self.vals = vals
    def rows(self, data) -> int:
        return max(data[0])+1

    def __call__(self, data, A: xp.ndarray) -> None:
        indices, d = data
        # calculate all bins up-front
        vals = self.bin(xp.asarray(d[self.vals]))
        for i,m,n in zip(indices, d[self.offsets], d[self.lengths]):
            A[i] = xp.bincount(vals[m:m+n], minlength=self.nbins)

def quantize(data: List[T], quantizers: List[Quantizer[T]]) -> xp.ndarray:
    """ Quantize a batch of samples.
    """
    rows = max( (q.rows(d) for d,q in zip(data, quantizers)), 0 )
    cols = sum( (q.nbins for q in quantizers), 0)
    A = xp.zeros((rows, cols))

    end = 0
    for d, q in zip(data, quantizers):
        start = end
        end += q.nbins
        q(d, A[:,start:end])
    return A

### Helper functions for scatterquantize

def mk_index(evts: List[np.ndarray]) -> Dict[int,int]:
    """ Create an index out of all the event numbers
    so that they are able to be numbered consecutively.
    """
    ans = set()
    for e in evts:
        ans.update( set(e) )
    # should we also count how many times each
    # event occurs in >1 detector?
    return {evt:i for i,evt in enumerate(ans)}

def create_indices(evts: List[np.ndarray]) -> List[List[int]]:
    idx = mk_index(evts)
    return [ [idx[i] for i in e] for e in evts ]

def scatterquantize(data: List[T], quantizers: List[Quantizer[T]],
            join_key='events',
        ) -> xp.ndarray:
    """Quantize a batch of samples with an EventPackedQuantizer

    Join on the join_key, read offsets from the offset key, etc.
    """
    indices = create_indices([d[join_key] for d in data])

    rows = max( (max(idx, default=0)+1 for idx in indices), default=0 )
    cols = sum( (q.nbins for q in quantizers), 0)
    A = xp.zeros((rows, cols))

    end = 0
    for idx, d, q in zip(indices, data, quantizers):
        start = end
        end += q.nbins
        q( (idx, d), A[:,start:end] )
    return A

"""
Note: we could use a custom kernel if A and B
were sparse-packed data structures (giving a list of the locations of "1"),

assert len(A) == len(B)
for k in range(len(A)):
    for i in A[k]:
        for j in B[k]:
            C[i,i] += 1
            C[i,j] += 1
            C[j,i] += 1
            C[j,j] += 1

Or, even more access-efficient, we could
skip the j,i combination and maintain the diagona
list separately:

    C[i,j] += 1
    diag[i] += 1
    diag[j] += 1


>>> x = cp.arange(6, dtype='f').reshape(2, 3)
>>> y = cp.arange(3, dtype='f')
>>> kernel = cp.ElementwiseKernel(
...     'float32 x, float32 y', 'float32 z',
...     '''
...     if (x - 2 > y) {
...       z = x * y;
...     } else {
...       z = x + y;
...     }
...     ''', 'my_kernel')
>>> kernel(x, y)
array([[ 0.,  2.,  4.],
       [ 0.,  4.,  10.]], dtype=float32)
"""

if __name__=="__main__":
    data = [None, None]
    quantizers = [Quantizer(2600, 4600, 1000),
                  Quantizer(2600, 4600, 1000)]
    ans = quantize(data, quantizers)
    print(ans)
    print(ans.shape)
