#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct
import cPickle, gzip
import numpy as np

def save_numpy_w2v(M, vocab, outfname):
    assert M.shape[0] == len(vocab)
    f = open(outfname, 'wb')
    vocSize, embSize = (M.shape[0], M.shape[1])
    print >> f, '{:d} {:d}'.format(M.shape[0], M.shape[1])
    for i in xrange(len(vocab)):
        f.write('{} '.format(vocab[i].encode('utf-8')))
        f.write(struct.pack('{:d}f'.format(embSize), *tuple(M[i])))
        f.write('\n')
    f.close()

def save_numpy_hidden(H, b, outfname):
    assert H.shape[1] == b.shape[0]
    f = open(outfname, 'wb')
    print >> f, '{:d} {:d}'.format(H.shape[0], H.shape[1])
    for i in xrange(H.shape[0]):
        f.write(struct.pack('{:d}f'.format(H.shape[1]), *tuple(H[i])))
    f.write(struct.pack('{:d}f'.format(b.shape[0]), *tuple(b)))
    f.close()

def save_numpy_output(H, b, outfname):
    f = open(outfname, 'wb')
    print >> f, '{:d}'.format(H.shape[0])
    f.write(struct.pack('{:d}f'.format(H.shape[0]), *tuple(H)))
    f.write(struct.pack('1f', b))
    f.close()
