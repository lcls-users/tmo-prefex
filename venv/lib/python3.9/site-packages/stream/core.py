from typing import (
    Optional,
    List,
    TypeVar,
    Generic,
    Union,
    #ParamSpec,
    #Concatenate,
)
from collections.abc import (
    Iterable,
    Iterator,
    Callable,
)
import inspect
import itertools
import functools

# Note on types: An Iterable has __iter__,
# while an Iterator has __next__ (and also __iter__ ~ self)
# so iter : Iterable[T] -> Iterator[T]
#
# This distinction is extremely important, since the
# stream processing functions all modify the stream in-place.
# iterators maintain state, so that __next__ keeps advancing.
# Iterables, on the other hand, start at the beginning of
# the stream in every consumer (e.g. `for i in iterable` loop).

# see https://docs.python.org/3/library/typing.html#typing.Concatenate
#P = ParamSpec('P')
S = TypeVar('S')
T = TypeVar('T')
R = TypeVar('R')
#SourceType = Iterable[S]
#StreamFunc = Callable[Concatenate[Iterator[S], P], Iterable[T]]
StreamFunc = Callable[..., Iterable[T]]
#SinkFunc   = Callable[Concatenate[Iterator[S], P], R]

#_____________________________________________________________________
# Base class


class BrokenPipe(Exception):
    pass

class BaseStream(Generic[R]):
    """A stream object which, when run, returns a result of type R.
    R may be a final value or a Stream or Sink (in case no Source is connected).

    The operator >> is a synonym for BaseStream.pipe.
    It "runs" the combined stream `a >> b`.

    Explicitly, the expression `a >> b` means
      - `b(iter(a)) if hasattr(a, '__iter__')`
      - `Stream(lambda it: b(a(it))) if isinstance(b, Stream)`
      - `Sink(lambda it: b(a(it)))` otherwise
    
    >>> [1, 2, 3] >> Sink(list)
    [1, 2, 3]
    >>> Source([1, 2, 3]) >> list
    [1, 2, 3]
    """
    @staticmethod
    def pipe(inp: 'Union[BaseStream[Iterable[S]], Iterable[S]]',
             out: 'Union[BaseStream[T], Callable[[Iterator[S]], T]]'
            ) -> T:
            #) -> 'Union[T, Stream[S,T], Sink[S,T]]':
        """Connect inp and out.  If out is not a Stream instance,
        it should be a sink (function callable on an iterable).

        Inp must be either an iterable (i.e. Source) or a Stream (i.e. pure).
        """

        assert hasattr(out, "__call__"), "Cannot compose a stream with a non-callable."
        if hasattr(inp, "__iter__"): # source stream
            return out(iter(inp))    # connect streams
        assert isinstance(inp, Stream), "Input to >> Stream/Sink should be a stream."

        # compose generates only these closures
        @functools.wraps(out)
        def close(iterator):
            return out(iter(inp(iterator)))

        if isinstance(out, Stream):
            return Stream(close) # type: ignore[return-value]
        return Sink(close) # type: ignore[return-value]

    def __rshift__(self, outpipe):
        return Stream.pipe(self, outpipe)

    def __rrshift__(self, inpipe):
        return Stream.pipe(inpipe, self)


class Source(Generic[S], Iterable[S], BaseStream[Iterable[S]]):
    """A Source is a BaseStream with a connected source.
    It represents a lazy list.  That is stored internally as
    the iterator attribute.

    It defines __iter__(self) for consumers to use.

    Source objects carry state, including the current
    read position and remaining items in the stream.
    So, they can be read from and append to.

    >>> s = Source(range(3))
    >>> s << range(-4, 0)
    Source(<itertools.chain object at ...>)
    >>> s >> list
    [0, 1, 2, -4, -3, -2, -1]
    """
    iterator : Iterator[S]

    def __init__(self, iterable : Iterable[S] = []) -> None:
        self.setup(iterable)

    def setup(self, iterable : Iterable[S]) -> None:
        self.iterator = iter(iterable)

    def __iter__(self) -> Iterator[S]:
        return self.iterator

    #def __reverse__(self) -> Iterator[S]:
    #    # Using this function is discouraged.
    #    return reverse(self.iterator)

    def __lshift__(self, more : Iterable[S]) -> 'Source[S]':
        self.extend(more)
        return self

    def extend(self, *other : Iterable[S]) -> None:
        self.setup( itertools.chain(self.iterator, *other) )

    #def __call__(self, iterator):
    #    #"""Append to the end of iterator."""
    #    return itertools.chain(iterator, self.iterator)

    def __repr__(self) -> str:
        return "Source(%s)" % repr(self.iterator)

class Stream(Generic[S,T], BaseStream[Iterable[T]]):
    """A stream is an iterator-processing function.
    When connected to a data source, it is also a lazy list.
    The lazy list is represented by a Source.
    
    The iterator-processing function is represented by the method
    __call__(iterator).  Since this method is only called
    when a source is actually defined, it always returns
    a Source object.
    
    The `>>` operator is defined from BaseStream.
    """
    def __init__(self,
                 fn : StreamFunc,
                 *args, # : P.args,
                 **kws, # : P.kwargs
                ) -> None:
        if not hasattr(fn, "__call__"):
            assert not hasattr(fn, "__iter__"), "Use Source() instead."
            assert hasattr(fn, "__call__"), "Stream function must be callable."
        self.fn = fn
        self.args = args
        self.kws = kws

    def __call__(self, iterator : Iterator[S]) -> Source[R]:
        """Consume the iterator to return a new Source (iterable)."""
        return Source(self.fn(iterator, *self.args, **self.kws))
        # Note: The function should be a generator, so this is equivalent to
        # yield from self.fn(gen, *self.args, **self.kws)

    def __repr__(self):
        if len(self.kws) > 0:
            return 'Stream(%s, *%s, **%s)' % (repr(self.fn),
                                            repr(self.args),
                                            repr(self.kws))
        if len(self.args) > 0:
            return 'Stream(%s, *%s)' % (repr(self.fn), repr(self.args))
        return 'Stream(%s)' % (repr(self.fn),)


class Sink(Generic[S, R], BaseStream[R]):
    def __init__(self, consumer, *args, **kws):
        self.consumer = consumer
        self.args = args
        self.kws = kws

    def __call__(self, iterator : Iterable[S]) -> R:
        """Consume the iterator to yield a final value."""
        return self.consumer(iterator, *self.args, **self.kws)

    def __repr__(self):
        if len(self.kws) > 0:
            return 'Sink(%s, *%s, **%s)' % (repr(self.consumer),
                                            repr(self.args),
                                            repr(self.kws))
        if len(self.args) > 0:
            return 'Sink(%s, *%s)' % (repr(self.consumer), repr(self.args))
        return 'Sink(%s)' % (repr(self.consumer),)


#_______________________________________________________________________
# Decorators for creating Source, Stream, and Sink instances.

def _single_arg_stream(func) -> bool:
    """Helper function to determine whether
    the given function only accepts a single argument.

    We use this to check whether we are decorating
    `f(iterator)` vs. `f(iterator, <more args>)` vs. `f()` [error]
    """
    params = inspect.signature(func).parameters
    # Count the number of positional-only parameters
    positional = 0
    for param in params.values():
        if param.kind  == inspect.Parameter.POSITIONAL_ONLY \
            or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional += 1
        else:
            return False
        # other types:
        #inspect.Parameter.KEYWORD_ONLY
        #inspect.Parameter.VAR_POSITIONAL
        #inspect.Parameter.VAR_KEYWORD
    if positional == 0:
        raise ValueError(f"Stream processing function {func} must accept an iterator as its first argument.")
    return positional == 1

def source(fn : Callable[..., Iterable[S]]) -> Callable[..., Source[S]]:
    """ Handy source decorator that wraps fn with a Source
    class so that it can be used in >> expressions.

    e.g.

    #>>> @source
    #>>> def to_source(rng):
    #>>>    yield from rng
    #>>> to_source(range(4, 6)) >> tuple
    #(4, 5)
    """
    @functools.wraps(fn)
    def gen(*args, **kws) -> Source[S]:
        return Source(fn(*args, **kws))
    return gen

def stream(fn : Callable[..., Iterable[T]]
          ) -> Callable[..., Stream[S,T]]:
#          ) -> Union[Stream[S,T], Callable[..., Stream[S,T]]]:
    """ Handy stream decorator that wraps fn with a Stream
    class so that it can be used in >> expressions.

    Basically, the first argument to the function becomes implicit.

    e.g.

    #>>> @stream
    #>>> def add_n(it : Iterator[int], n : int):
    #>>>    for i in it:
    #>>>        yield i+n
    #>>> [1] >> add_n(7) >> tuple
    #(8,)
    """
    if _single_arg_stream(fn):
        return Stream(fn) # type: ignore[return-value]

    @functools.wraps(fn)
    def gen(*args, **kws):
        return Stream(fn, *args, **kws)
    return gen

def sink(fn : Callable[..., R]
        ) -> Callable[..., Sink[S,R]]:
#       ) -> Union[Sink[S,R], Callable[..., Sink[S,R]]]:
    """ Handy stream decorator that wraps fn with a Stream
    class so that it can be used in >> expressions.

    Basically, the first argument to the function becomes implicit.

    e.g.

    #>>> @stream
    #>>> def add_n(it : Iterator[int], n : int):
    #>>>    for i in it:
    #>>>        yield i+n
    #>>> [1] >> add_n(7) >> tuple
    #(8,)
    """
    if _single_arg_stream(fn):
        return Sink(fn) # type: ignore[return-value]

    @functools.wraps(fn)
    def gen(*args, **kws) -> Sink[S,R]:
        return Sink(fn, *args, **kws)
    return gen


#_____________________________________________________________________
# main


if __name__ == "__main__":
    import doctest
    if doctest.testmod()[0]:
        import sys
        sys.exit(1)
