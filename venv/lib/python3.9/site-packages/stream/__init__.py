"""Lazily-evaluated, parallelizable pipeline.

Overview
========

Streams are iterables with a pipelining mechanism to enable
data-flow programming and easy parallelization.

The idea is to create a `Stream` class with 4 states: 
 - "pure" (no source or sink connected) 
 - "consumer" (sink but no source) 
 - "live" (source but no sink) 
 - "final" (source and sink connected) 

Both "pure" and "consumer" Stream-s maintain no internal
state.  They can be re-used with multiple input streams
to make different results.

Sources can be any iterable.  Sinks can be any callable.
When a source is connected to a sink, it becomes "final".
The sink (or its `__call__` method) will be called, with
the source as its argument.
The sink should iterate over the source
-- advancing the state of the source.
The "final" result evaluates to the return value of the sink.

Data sources can be created with `Source(iterable)`
pure transformers can be created with `Stream(generator, *args, **kws)`,
(where generator does not have an `__iter__` method),
and sinks are any callable taking a stream as an argument.
If you want to repeatedly call `callable(x, *args, **kws)` for each
element in the stream, there is a helper, `SinkFn(callable, *args, **kws)`.

If the sink does not completely consume
the source, then the source can be connected to further
sinks to yield the remaining values from the stream.

Sources and sinks can be distinguished by whether they contain
`__iter__` or `__call__` methods.  Only "live" sources define
`__iter__`.  All Streams and Sinks define `__call__`.  However, calling
a "pure" stream returns another Stream, whereas calling
a "sink" consumes the input and returns the result.

The idea is to take the output of a function that turn an iterable into
another iterable and plug that as the input of another such function.
While you can already do this using function composition, this package
provides an elegant notation for it by overloading the '>>' operator.

This approach focuses the programming on processing streams of data, step
by step.  A pipeline usually starts with a producer, then passes through
a number of filters.  Multiple streams can be branched and combined.
Finally, the output is fed to an accumulator, which can be any function
of one iterable argument.

Producers:  anything iterable
    + from this module:  seq, gseq, repeatcall, chaincall

Filters:
    + by index:  take, drop, takei, dropi
    + by condition:  filter, takewhile, dropwhile
    + by transformation:  apply, map, fold
    + by combining streams:  prepend, tap
    + for special purpose:  chop, cut, flatten

Accumulators:  item, maximum, minimum, reduce
    + from Python:  list, sum, dict, max, min ...
    (anything you can call with an iterable)

Values are computed only when an accumulator forces some or all evaluation
(not when the stream are set up).


Parallelization
===============

All parts of a pipeline can be parallelized using multiple threads or processes.

When a producer is doing blocking I/O, it is possible to use a ThreadedFeeder
or ForkedFeeder to improve performance.  The feeder will start a thread or a
process to run the producer and feed generated items back to the pipeline, thus
minimizing the time that the whole pipeline has to wait when the producer is
blocking in system calls.

If the order of processing does not matter, an ThreadPool or ProcessPool
can be used.  They both utilize a number of workers in other theads
or processes to work on items pulled from the input stream.  Their output
are simply iterables respresented by the pool objects which can be used in
pipelines.  Alternatively, an Executor can perform fine-grained, concurrent job
control over a thread/process pool.

Multiple streams can be piped to a single PCollector or QCollector, which
will gather generated items whenever they are avaiable.  PCollectors
can collect from ForkedFeeder's or ProcessPool's (via system pipes) and
QCollector's can collect from ThreadedFeeder's and ThreadPool's (via queues).
PSorter and QSorter are also collectors, but given multiples sorted input
streams (low to high), a Sorter will output items in sorted order.

Using multiples Feeder's and Collector's, one can implement many parallel
processing patterns:  fan-in, fan-out, many-to-many map-reduce, etc.


Articles
========

Articles written about this module by the author can be retrieved from
<http://blog.onideas.ws/tag/project:stream.py>.

* [SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html)
* [pointfree](https://github.com/mshroyer/pointfree)
* [Generator Tricks for Systems Programmers](http://www.dabeaz.com/generators/Generators.pdf)
"""

#import pkg_resources
#try:
#    __version__ = pkg_resources.get_distribution(__package__).version
#except Exception:
#    __version__ = 'unknown'
from importlib import metadata
try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = 'unknown'

from .core import Source, Stream, Sink, source, stream, sink
from .ops import (
    take, last, takei, drop, dropi,
    apply, map, filter, takewhile, dropwhile,
    fold, chop, cut,
    seq, gseq, repeatcall, chaincall, sorter,
    item, maximum, minimum,
    reduce, flatten, prepend,
    append, dup, tap, tee,
)
# Note Pipe is only used by ForkedFeeder and PCollector/PSorter
from .parallel import (
    iterqueue, QueueSource, QueueSink, sink_cb,
    ThreadStream, ProcessStream
)
