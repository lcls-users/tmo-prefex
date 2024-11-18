import itertools
from collections.abc import Iterator

from stream import Stream, Source, stream, take

@stream
def xmap(items, fn, *args, **kws):
    """ map with some fixed args
    first arg will be stream element
    """
    for i in items:
        yield fn(i, *args, **kws)

def split(*streams : Stream):
    """ Takes a list of streams and applies each to the
    same input value.  The output of this stream will
    be a zip() over them -- so tuples of outputs, 1 per
    stream.

    Note that if the individual streams don't yield in-sync,
    then the inputs will pile up into a buffer.
    """
    def fn(iterator : Iterator) -> Source:
        branches = itertools.tee(iterator, len(streams))
        # form a list of iterators by running each stream
        # on each branch of the input
        its = [s(x) for s,x in zip(streams, branches)]
        #yield from zip(*its)
        return Source( zip(*its) )
    return Stream(fn)

@stream
def variable_chunks(iterator, sizes: Iterator[int]):
    """Yield lists of items from the iterator with variable
    sizes.  If sizes reaches its end, this stream ends.
    
    If you want an infinite stream, use something like:

    >>> sizes = stream.Source([1<<4, 1<<6, 1<<8, 1<<10]) << stream.seq(1<<13,0)

    which will stick at 8192 forever... about one second at 8 kHz
    """
    n = 0
    for n in sizes:
        s = iterator >> take(n) >> list
        if s:
            yield s
        else:
            return
    while n > 0:
        s = iterator >> take(n) >> list
        if s:
            yield s
        else:
            return

def test_split():
    import stream

    state = [] # track execution of src with this
    def src():
        for i in range(4):
            print(f"at {i}")
            state.append(i)
            yield i
    s = src()
    s >>= stream.map(lambda x: x)
    j = 0
    for i in s >> split(stream.map(lambda x: 'a'),
                        stream.map(lambda x: x+1),
                        stream.map(lambda x: 2*x)):
        #print(i)
        assert tuple(state) == tuple(range(j+1))
        assert i == ('a', j+1, j*2)
        j += 1

def test_variable():
    import stream
    expander = variable_chunks([2, 5, 10])
    for i, v in enumerate(stream.seq() >> expander >> take(4)):
        print(v)
        assert isinstance(v, list)
        if i == 0:
            assert tuple(v) == tuple(range(2))
        elif i == 1:
            assert tuple(v) == tuple(range(2, 2+5))
        elif i == 2:
            assert tuple(v) == tuple(range(2+5, 2+5+10))
        elif i == 3:
            assert tuple(v) == tuple(range(2+5+10, 2+5+10*2))
        else:
            assert False

if __name__=="__main__":
    #test_split()
    test_variable()
