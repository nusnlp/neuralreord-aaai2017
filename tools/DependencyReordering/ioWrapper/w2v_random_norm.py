#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, os
import cPickle, gzip
import numpy as np

def listFromFile(fname):
    ret = []
    f = open(fname, 'r')
    for l in f:
        ret.append(l.decode('utf-8').strip())
    f.close()
    return ret

if __name__ == '__main__':
    rng = np.random.RandomState(1234)

    # Read word2vec binary file
    print >> sys.stderr, "Loading vocabulary from %s" % sys.argv[1]
    allVcb = listFromFile(sys.argv[1])
    embSize = int(sys.argv[2])
    outName = sys.argv[3]
    alpha = float(sys.argv[4])

    rndhi = alpha
    rndlo = -rndhi
    
    totalVcbSize = len(allVcb)
    print >> sys.stderr, "Random initialization started..."
    M = np.memmap('{}.mat.mmap'.format(outName), dtype='float32', mode='w+', shape=(totalVcbSize, embSize))
    M[:] = rng.uniform(rndlo, rndhi, (totalVcbSize, embSize))
    print >> sys.stderr, "DONE! Writing to file %s" % outName
    
    # Close the file and pickle embedding
    M.flush()
    del M
