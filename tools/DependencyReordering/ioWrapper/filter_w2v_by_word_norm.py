#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, re
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
    rng = np.random.RandomState(1234)

    # Open file and read header information
    loader = W2VLoader(sys.argv[1])
    vcbName = sys.argv[2]
    outName = sys.argv[3]
    alpha = float(sys.argv[4])

    # optional suffixes
    suffix_reStr = r'(' + '|'.join(sys.argv[5].decode('utf-8').split()) + ')$' if len(sys.argv) > 5 else ''
    suffix_regex = re.compile(suffix_reStr)

    rndhi = alpha
    rndlo = -rndhi

    usedVcb = listFromFile(vcbName)
    embTab = []
    for v in usedVcb:
        if suffix_regex.search(v):
            v_s = suffix_regex.sub('', v)  # stripped of allowed suffix
            try:
                embTab.append(loader.M[loader.wordID[v_s]])
            except KeyError:
                embTab.append(rng.uniform(rndlo, rndhi, (loader.embSize,)))
        else:
            try:
                embTab.append(loader.M[loader.wordID[v]])
            except KeyError:
                embTab.append(rng.uniform(rndlo, rndhi, (loader.embSize,)))
        print v.encode('utf-8')

    M = np.memmap('{}.mat.mmap'.format(outName), dtype='float32', mode='w+', shape=(len(embTab), loader.embSize))
    M[:] = np.asarray(embTab)
    
    # Close the file and pickle embedding
    M.flush()
    del M
