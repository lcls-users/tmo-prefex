import itertools
from collections.abc import Iterator

from stream import Stream, Source, stream

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


if __name__=="__main__":
    import stream
    def src():
        for i in range(4):
            print(f"at {i}")
            yield i
    s = src()
    s >>= stream.map(lambda x: x)
    for i in s >> split(stream.map(lambda x: 'a'),
                        stream.map(lambda x: x+1),
                        stream.map(lambda x: 2*x)):
        print(i)

