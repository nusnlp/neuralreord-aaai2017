#!/usr/bin/python
import sys, re
import argparse
from dependency_input import Dependency
from operator import itemgetter
import numpy as np

def smart_open(fname, mode = 'r'):
    if fname.endswith('.gz'):
        import gzip
        # Using max compression (9) by default seems to be slow.
        # Let's try using the fastest.
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("argv", metavar='A', nargs='+')
    args = argparser.parse_args()

    sampleSize = int(args.argv[0])
    dimSize = int(args.argv[1])
    trainmapName = args.argv[2]
    print >> sys.stderr, str(sampleSize), str(dimSize), trainmapName

    dotCutoff = 100000
    dpl = 100
    nlCutoff = dotCutoff * dpl
    lCount = 0

    np.random.seed(1234)

    print >> sys.stderr, "Loading training instances"
    fp = np.memmap(trainmapName, mode='w+', dtype='int32', shape=(sampleSize+1, dimSize))
    fp[0,0] = sampleSize
    fp[0,1] = dimSize
    lCount = 0
    for line in sys.stdin:
        line = line.decode('utf-8').strip()
        tokens = line.split()
        try:
            fp[lCount+1,:-1] = [int(t) for t in tokens[1:]]
            fp[lCount+1,-1] = int(tokens[0])
        except KeyError:
            continue
        # trace progress
        lCount += 1
        if lCount % dotCutoff == 0:
            sys.stderr.write('.')
        if lCount % nlCutoff == 0:
            print >> sys.stderr, "%d" % lCount
    print >> sys.stderr, "[%d instances in total]" % lCount
    print >> sys.stderr, "Shuffling training instances started ..."
    np.random.shuffle(fp[1:])
    print >> sys.stderr, "Shuffling training instances done, now dumping ..."
    fp.flush()
    del fp
    print >> sys.stderr, "COMPLETED!"
