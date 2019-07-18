#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct
import cPickle, gzip
import numpy as np
from w2v_to_numpy import W2VLoader

def listFromFile(fname):
    ret = []
    f = open(fname, 'r')
    for l in f:
        l = l.decode('utf-8').strip()
        ret.append(l)
    f.close()
    return ret

if __name__ == '__main__':

    # Open file and read header information
    loader = W2VLoader(sys.argv[1])
    vcbName = sys.argv[2]
    outName = sys.argv[3]

    usedVcb = listFromFile(vcbName)
    embTab = []
    for v in usedVcb:
        try:
            embTab.append(loader.M[loader.wordID[v]])
            print v.encode('utf-8')
        except KeyError:
            continue

    M = np.memmap('{}.mat.mmap'.format(outName), dtype='float32', mode='w+', shape=(len(embTab), loader.embSize))
    M[:] = np.asarray(embTab)
    
    # Close the file and pickle embedding
    M.flush()
    del M
