#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, os
import cPickle, gzip
import numpy as np
from w2v_to_numpy import W2VLoader

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
    loader = W2VLoader(sys.argv[1])
    suppVcb = listFromFile(sys.argv[2])
    outName = sys.argv[3]
    
    totalVcbSize = loader.vocSize + len(suppVcb)
    M = np.memmap('{}.mat.mmap'.format(outName), dtype='float32', mode='w+', shape=(totalVcbSize, loader.embSize))
    M[loader.vocSize:] = rng.uniform(-0.5 / loader.embSize, 0.5 / loader.embSize, (len(suppVcb), loader.embSize))
    M[:loader.vocSize] = loader.M[:]
    for w in loader.vocab:
        print w
    for w in suppVcb:
        print w.encode('utf-8')
    
    # Close the file and pickle embedding
    M.flush()
    del M
