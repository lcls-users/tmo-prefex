""" Parallel patterns are created by
    sending stream output to pipes / queues / sockets
    and receiving stream inputs from same.

    The best IPC channels to use are Queue-s --
    queue.Queue or queue.SimpleQueue[python 3.7+]
    for threads and multiprocessing.Queue or
    multiprocessing.SimpleQueue for processes.

    These allow setting queue maximums to prevent evaluating
    the source stream faster than the receiver can process it.
"""
from typing import (
    Optional,
    TypeVar,
    Union,
    #ParamSpec,
    #Concatenate,
)
from collections.abc import (
    Iterable,
    Iterator,
    Callable,
)
from contextlib import contextmanager

import itertools
import heapq
import queue

import threading
import time

_map = map

from .core import Source, Stream, Sink, source, stream, sink
from .ops import map

#try:
import multiprocessing
_nCPU = multiprocessing.cpu_count()
AnyQueue = Union[queue.Queue, multiprocessing.SimpleQueue]
#except ImportError:
#    _nCPU = 1

S = TypeVar('S')
T = TypeVar('T')


#___________
# Sources

def _try_stop(queue : AnyQueue) -> None:
    # Re-broadcast, in case there is another listener blocking on
    # queue.get().  That listener will receive StopIteration and
    # re-broadcast to the next one in line.
    try:
        queue.put(StopIteration)
    except IOError:
        # Could happen if the Queue is based on a system pipe,
        # and the other end was closed.
        pass

@source
def iterqueue(queue : AnyQueue) -> Iterator:
    """Turn a (queue/multiprocessing).(Queue/SimpleQueue)
    into an thread-safe iterator which will exhaust when StopIteration is
    put into it.
    
    This simple queue source works with one queue only.
    QueueSource handles the multi-queue case, falling back
    to this for just 1 queue.
    """
    while True:
        item = queue.get()
        if item is StopIteration:
            _try_stop(queue)
            break
        yield item


@source
def QueueSource(*inqueues : AnyQueue, waittime : float = 0.1) -> Iterator:
    """Collect items from many ThreadedSource's or ThreadStream's.
    
    All input queues are polled individually.  When none is ready, the
    collector sleeps for a fix duration before polling again.

    Params:
        waitime: the duration that the collector sleeps for
                 when all input pipes are empty

    >>> q1 = queue.Queue(12)
    """
    queues = list(inqueues)
    #if len(queues) == 1: # would return Sink, not iterator...
    #    return iterqueue(queues[0])

    def gen(queues):
        while queues:
            ready = [q for q in queues if not q.empty()]
            if not ready:
                time.sleep(waittime)
            for q in ready:
                item = q.get()
                if item is StopIteration:
                    _try_stop(q)
                    queues.pop(queues.index(q))
                else:
                    yield item
    return gen(queues)

#_____________________________________________________________________
# Threaded/forked feeder

@sink
def QueueSink(iterator: Iterable[S], outqueue: AnyQueue) -> None:
    """Eagerly evaluate the stream and send it to a queue.

    Note: generator, args, and kwargs are all pickled
    and sent to the thread for processing, so this will only
    speedup the actual operations done during stream generation,
    not creation of the stream.
    
    Params:
        outqueue: queue where output will be put()

    TODO: Add arg to execute in separate thread.
          This should improve performance when the generator often
          blocks in system calls.

    >>> from stream.ops import last
    >>> outqueue = queue.Queue(0) # note: <3 will deadlock
    >>> ["test"]*3 >> QueueSink(outqueue) # blocks if queue is full
    >>> iterqueue(outqueue) >> last()
    'test'
    >>> 
    >>> QueueSource(outqueue) >> last()
    Traceback (most recent call last):
      ...
    IndexError: list index out of range
    """
    for x in iterator:
        outqueue.put(x)
    outqueue.put(StopIteration)

def _run_receiver(q : AnyQueue, recv : Sink) -> None:
    # TODO: return this and do something with it at callback completion?
    QueueSource(q) >> recv

@contextmanager
def sink_cb(recv : Sink[S, T],
            maxsize=1024) -> Iterator[Callable[[S],None]]:
    """ Transform the Sink to a callable within a context.
    The sink will run in a separate thread, and pull
    from a queue that is populated only when the stream advances.

    Params:
        recv: sink receiving data. Its return value is lost.
        maxsize: maximum queue size before blocking on callback()
                 to wait for the accumulated callbacks to be consumed.

    >>> from stream.ops import tee, append, take
    >>> ans = []
    >>> tapped = "cookie" >> tee(sink_cb, append(ans), maxsize=8)
    >>> tapped >> take(4) >> "".join
    'cook'
    >>> ans # not available yet
    []
    >>> tapped >> list
    ['i', 'e']
    >>> ans
    ['c', 'o', 'o', 'k', 'i', 'e']
    >>> tapped << "."
    Source(<itertools.chain object at ...>)
    >>> tapped >> "".join
    '.'
    >>> ''.join(ans) # FIXME should be 'cookie.', but termination is tricky
    'cookie'
    """
    # Create a message queue.
    q : queue.Queue = queue.Queue(maxsize)
    # Run the receiving process in a separate thread.
    # Don't wait for it to terminate when program ends.
    t = threading.Thread(target=_run_receiver,
                         args=(q,recv),
                         daemon=True)
    t.start()
    # Setup the callback to send to the queue.
    def callback(item : S) -> None:
        q.put(item)
    yield callback
    _try_stop(q) # signal the thread to shutdown
    t.join() # wait for queue to consume all input


#_____________________________________________________________________
# Asynchronous stream processing using a pool of threads or processes

@stream
def parallel(iterator : Iterator[S],
             worker_stream : Stream[S,T],
             typ="thread", poolsize=_nCPU) -> Iterable[T]:
    """
    Stream combinator taking a worker_stream and executing it in parallel.

    Params:
        worker_stream: a Stream object to be run on each thread/process
        typ: either "thread" or "process" indicating whether to use
             thread or process-based parallelism

    >>> range(10) >> ThreadStream(map(lambda x: x*x)) >> sum
    285
    >>> range(10) >> ProcessStream(map(lambda x: x*x)) >> sum
    285
    """

    inqueue : AnyQueue
    outqueue : AnyQueue
    if typ == "thread":
        inqueue  = queue.Queue()
        outqueue = queue.Queue()
        recvfrom = iterqueue
        start = threading.Thread
    else:
        inqueue  = multiprocessing.SimpleQueue()
        outqueue = multiprocessing.SimpleQueue()
        recvfrom = iterqueue
        start = multiprocessing.Process # type: ignore[assignment]
    
    #failqueue = Qtype()
    #failure = Source(recvfrom(failqueue))
    def work():
        try:
            for ans in (recvfrom(inqueue) >> worker_stream):
                #yield ans
                outqueue.put(ans)
        except Exception as e:
            #failqueue.put((next(dupinput), e))
            #outqueue.put(e)
            raise
    workers = []
    for _ in range(poolsize):
        t = start(target=work)
        workers.append(t)
        t.start()
    def cleanup():
        # Wait for all workers to finish,
        # then signal the end of outqueue and failqueue.
        for t in workers:
            t.join()
        outqueue.put(StopIteration)
    cleaner_thread = threading.Thread(target=cleanup)
    cleaner_thread.start()
    
    def feed():
        for item in iterator:
            inqueue.put(item)
        inqueue.put(StopIteration)
    feeder_thread = threading.Thread(target=feed)
    feeder_thread.start()

    yield from iterqueue(outqueue)

    feeder_thread.join()
    cleaner_thread.join()

def ThreadStream(worker_stream : Stream[S,T], poolsize=_nCPU*4) -> Stream[S,T]:
    return parallel(worker_stream, typ="thread", poolsize=poolsize)

def ProcessStream(worker_stream : Stream[S,T], poolsize=_nCPU) -> Stream[S,T]:
    return parallel(worker_stream, typ="process", poolsize=poolsize)

