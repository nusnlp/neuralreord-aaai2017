#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, os
import cPickle, gzip
import numpy as np
from w2v_to_numpy import W2VLoader

if __name__ == '__main__':

    # Open numpized word2vec file
    loader = W2VLoader(sys.argv[1])
    vcbIdx = {}
    for i, v in enumerate(loader.vocab):
        vcbIdx[v] = i
        
    # console to see vector    
    print >> sys.stderr, 'Ready to receive input...'
    line = sys.stdin.readline()
    while len(line) != 0:
        rline = line.strip()
        try:
            print loader.M[vcbIdx[rline]]
        except KeyError:
            print >> sys.stderr, 'ERROR: Cannot find word'

        sys.stdout.flush()
        line = sys.stdin.readline()

