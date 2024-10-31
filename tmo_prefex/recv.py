import stream
from nng import puller, rate_clock, clock0

def main(argv):
    assert len(argv) == 2, f"Usage: {argv[0]} <addr>"

    addr = argv[1]
    clock = stream.fold(rate_clock, clock0())

    for stat in puller(addr, 1) >> stream.map(len) >> clock:
        stat['name'] = 'recv'
        print(stat)

if __name__=="__main__":
    import sys
    main(sys.argv)
