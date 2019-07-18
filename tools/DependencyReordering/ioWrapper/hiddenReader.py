#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, os
import cPickle, gzip
import numpy as np

class HiddenLoader:
    def __init__(self, fname):
        f = open(fname, 'rb')
        header = f.readline().decode('utf-8').strip()
        self.rows, self.cols = tuple([int(n) for n in header.split()])
        self.M = np.zeros((self.rows, self.cols), dtype='float32')
        print >> sys.stderr, "Row size = {:d}; Column size = {:d}".format(self.rows, self.cols)

        # Read line by line
        for r in xrange(self.rows):
            self.M[r] = struct.unpack('{:d}f'.format(self.cols), f.read(4 * self.cols))

        self.bias = np.zeros((self.cols,), dtype='float32')
        self.bias = struct.unpack('{:d}f'.format(self.cols), f.read(4 * self.cols))
        f.close()

if __name__ == '__main__':

    # Open file and read header information
    loader = HiddenLoader(sys.argv[1])
    outFile = sys.stdout
    print >> outFile, 'Weights:'
    for r in xrange(loader.rows):
        print >> outFile, ' '.join([str(x) for x in loader.M[r]])

    print >> outFile, 'bias:'
    print >> outFile, ' '.join([str(x) for x in loader.bias])
    
