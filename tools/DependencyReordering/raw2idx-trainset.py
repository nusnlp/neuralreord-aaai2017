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
    argparser.add_argument("--with_others", dest="with_others", action='store_true', help="Include other features other than word pair links.")
    argparser.add_argument("argv", metavar='A', nargs='+')
    args = argparser.parse_args()

    vocabName = args.argv[0]
    inName = args.argv[1]
    outName = args.argv[2]
    vocabFile = smart_open(vocabName, 'r')
    inFile = smart_open(inName, 'r')
    outFile = smart_open(outName, 'w')

    dotCutoff = 10000
    dpl = 100
    nlCutoff = dotCutoff * dpl

    print >> sys.stderr, "Loading vocabulary ..."
    word2Idx = {}
    lCount = 0
    for w in vocabFile:
        idx = len(word2Idx)
        word2Idx[w.decode('utf-8').strip()] = idx
        # trace progress
        lCount += 1
        if lCount % dotCutoff == 0:
            sys.stderr.write('.')
        if lCount % nlCutoff == 0:
            print >> sys.stderr, "%d" % lCount
    word2Idx['<NULL>'] = -1 # sentinel

    dotCutoff = 100000
    nlCutoff = dotCutoff * dpl
    
    print >> sys.stderr, "Loading training instances"
    lCount = 0
    for line in inFile:
        line = line.decode('utf-8').strip()
        tokens = line.split()
        try:
            print >> outFile, "%s %s" % (tokens[0], ' '.join([str(word2Idx[t]) for t in tokens[1:]]))
        except KeyError:
            print >> sys.stderr, "%s %s" % (tokens[0].encode('utf-8'), ' '.join(tokens[1:]).encode('utf-8'))
            continue
        # trace progress
        lCount += 1
        if lCount % dotCutoff == 0:
            sys.stderr.write('.')
        if lCount % nlCutoff == 0:
            print >> sys.stderr, "%d" % lCount
    print >> sys.stderr, "[%d instances in total]" % lCount
    print >> sys.stdout, "%d" % lCount
