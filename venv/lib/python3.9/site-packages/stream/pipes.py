from collections.abc import Iterator, Iterable
import select
import sys
import time
import multiprocessing
from multiprocessing.connection import Connection
Pipe = multiprocessing.Pipe

from .core import source, sink

@source
def iterrecv(pipe : Connection) -> Iterable:
    """Turn a the receiving end of a multiprocessing.Connection object
    into an iterator which will exhaust when StopIteration is
    put into it.  _iterrecv is NOT safe to use by multiple threads.

    This handles the single-pipe case.  PipeSource handles
    the multiple input case, falling back to this for a single
    source.
    """
    while True:
        try:
            item = pipe.recv()
        except EOFError:
            break
        else:
            if item is StopIteration:
                break
            else:
                yield item


@source
def PipeSource(*inpipes : Connection, waittime : float = 0.0) -> Iterator:
    """Collect items from many ProcessSource's or ProcessStream's.
    """
    pipes = list(inpipes)
    #if len(pipes) == 1:
    #    return iterrecv(pipes[0])

    def gen(pipes):
        while pipes:
            ready, _, _ = select.select(pipes, [], [])
            for inpipe in ready:
                item = inpipe.recv()
                if item is StopIteration:
                    pipes.pop(pipes.index(inpipe))
                else:
                    yield item
    return gen(pipes)

@source
def _PipeSource(*inpipes : Connection, waittime : float = 0.1) -> Iterator:
    """Collect items from many ProcessSource's or ProcessStream's.

    All input pipes are polled individually.  When none is ready, the
    collector sleeps for a fix duration before polling again.

    Params:
        waitime: the duration that the collector sleeps for
                 when all input pipes are empty
    """
    pipes = list(inpipes)
    #if len(pipes) == 1:
    #    return iterrecv(pipes[0]) # would return Sink, not iterator

    def gen(pipes):
        while pipes:
            ready = [p for p in pipes if p.poll()]
            if not ready:
                time.sleep(waittime)
            for inpipe in ready:
                item = inpipe.recv()
                if item is StopIteration:
                    pipes.pop(pipes.index(inpipe))
                else:
                    yield item
    return gen(pipes)

if sys.platform == "win32":
    PipeSource = _PipeSource # no select on win32


# TODO: exchange data using shared memory
# TODO: read ACK-s from the receiver to guard against sending too much data to a slow receiver
@sink
def PipeSink(generator : Iterator, outpipe, maxsize=0) -> None:
    """Pass the stream to a pipe.
    """
    #outpipe, inpipe = Pipe(duplex=False)
    for x in generator:
        outpipe.send(x)
    outpipe.send(StopIteration)
