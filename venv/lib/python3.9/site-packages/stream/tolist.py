from typing import Union, List
from collections.abc import Iterator

from .core import Source, Stream, Sink, S
from .ops import take, drop, tap, last

class _SleepTaker:
    """Slice the input stream, return a list.

    Note: Negative start/end indices will force
    evaluating the entire stream.

    It is almost always better to use last(n)
    to retrieve the last n items of the stream instead.

    >>> i = Source(range(100)) # note i = range(100) starts at 0 each time...
    >>> i >> tolist[:10:2]
    [0, 2, 4, 6, 8]
    >>> i >> tolist[:5]
    [10, 11, 12, 13, 14]
    >>> range(20) >> tolist[::-2]
    [19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
    >>> range(20) >> tolist[3:-4:4]
    [3, 7, 11, 15]
    """
    @staticmethod
    def __getitem__(key) -> Sink[S, Union[S, List[S]]]:
        assert isinstance(key, (int, slice)), 'key must be an integer or a slice'
        if isinstance(key, int):
            if key < 0:
                return last(key) # type: ignore[return-value]
            return drop(key) >> next

        # slices are extremely messed up because of the off-by-one
        # error.
        def run(it : Iterator[S]) -> Union[S, List[S]]:
            step = 1 if key.step is None else key.step
            assert step != 0, 'Invalid step.'

            s = Source(it)
            n = 0
            def counter(x):
                nonlocal n
                n += 1

            if step > 0:
                a = 0 if key.start is None else key.start
                b = key.stop
                # seek to a
                if a < 0:
                    ans = s >> tap(counter) >> last(-a)
                    if b is None:
                        return ans[::step]
                    if b > 0: # adjust end-offset
                        b = b-n+len(ans)
                        if b < 0:
                            return []
                    return ans[:b:step]
                elif a > 0:
                    s = s >> tap(counter) >> drop(a)
                # handle stopping at b
                if b is None:
                    return (s >> list)[::step]
                if b > 0: # adjust end-offset
                    b = b-a
                    if b <= 0:
                        return []
                    return (s >> take(b) >> list)[::step]
                # b < 0
                ans = s >> list
                return ans[:b:step]
                        
            else: # stepping backwards
                # add one to change inclusion to leftwards.
                if key.stop == -1: # stupid case
                    return []
                a = 0 if key.stop is None else key.stop+1
                b = None if key.start is None else key.start+1
                if a < 0:
                    ans = s >> tap(counter) >> last(-a)
                    if b is None:
                        return ans[:key.stop:step]
                    if b > 0: # adjust end-offset
                        b = b-n+len(ans)
                        if b <= 0:
                            return []
                    return ans[b-1:key.stop:step]
                elif a > 0:
                    s = s >> tap(counter) >> drop(a)
                # end is now None (start of list)

                # handle stopping at b
                if b is None:
                    return (s >> list)[::step]
                if b > 0:  # adjust end-offset
                    b = b-a
                    if b <= 0:
                        return []
                    return (s >> take(b) >> list)[::step]
                # b < 0
                ans = s >> list
                return ans[b-1::step]

        return Sink(run)

    def __repr__(self):
        return '<tolist at %s>' % hex(id(self))

tolist = _SleepTaker()
