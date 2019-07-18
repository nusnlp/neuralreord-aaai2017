#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, os
import cPickle, gzip
import numpy as np

class LogreLoader:
    def __init__(self, fname):
        f = open(fname, 'rb')
        header = f.readline().decode('utf-8').strip()
        self.dim = int(header)
        self.M = np.zeros((self.dim,), dtype='float32')
        print >> sys.stderr, "Dimension = {:d}".format(self.dim)

        # Read line by line
        self.M = struct.unpack('{:d}f'.format(self.dim), f.read(4 * self.dim))
        self.bias = struct.unpack('1f', f.read(4))
        f.close()

if __name__ == '__main__':

    # Open file and read header information
    loader = LogreLoader(sys.argv[1])
    outFile = sys.stdout
    print >> outFile, 'Weights:'
    print >> outFile, ' '.join([str(x) for x in loader.M])

    print >> outFile, 'bias:'
    print >> outFile, ' '.join([str(x) for x in loader.bias])
    
