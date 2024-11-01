import random
from stream import source, take, fold
from lclstream.nng import pusher, clock0, rate_clock

@source
def arrays():
    while True:
        yield random.randbytes(1024**2) # 1M

def main(argv):
    assert len(argv) == 2, f"Usage: {argv[0]} <addr>"

    addr = argv[1]
    pipe = arrays() >> take(5) >> pusher(addr, 0) \
            >> fold(rate_clock, clock0())
    for stat in pipe:
        stat['name'] = 'send'
        print(stat)

if __name__=="__main__":
    import sys
    main(sys.argv)
