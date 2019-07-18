#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: w2v_to_numpy.py
# Convert binary W2V file to
import sys, struct, os
import cPickle, gzip
import numpy as np

class W2VLoader:
    def __init__(self, fname):
        # Open file and read header information
        if os.path.isfile(fname + '.npemb.mat.mmap') and os.path.isfile(fname + '.vcb'):
            self.vocab = []
            self.wordID = {}

            # Load vocabulary
            print >> sys.stderr, 'Loading prepared vocabulary ...'
            f = open(fname + '.vcb', 'r')
            for l in f:
                l = l.decode('utf-8', 'ignore').strip()
                self.vocab.append(l.encode('utf-8'))
                self.wordID[l] = len(self.vocab) - 1
            f.close()
            print >> sys.stderr, 'Loading memory map'
            self.M = np.memmap('{}.npemb.mat.mmap'.format(fname), dtype='float32', mode='r')
            self.M = self.M.reshape((len(self.vocab), self.M.shape[0] / len(self.vocab)))
            self.vocSize, self.embSize = self.M.shape
        else:
            f = open(fname, 'rb')
            header = f.readline().decode('utf-8').strip()
            vocSize, embSize = tuple([int(n) for n in header.split()])
            self.M = np.zeros((vocSize, embSize), dtype='float32')
            print >> sys.stderr, "Vocab size = {:d}; dimension = {:d}".format(vocSize, embSize)
            self.vocSize, self.embSize = (vocSize, embSize)
            self.vocab = []
            self.wordID = {}

            # Read line by line
            lCount = 0
            dotCut, dpl = (5000, 100)
            nlCut = dpl * dotCut
            for v in xrange(vocSize):
                charArr = ['']
                a = 0
                while True:
                    charArr[a] = f.read(1)
                    if charArr[a] is None or charArr[a] == ' ':
                        break
                    if charArr[a] != '\n':
                        a += 1
                        charArr.append('')
                del charArr[-1]
                self.vocab.append(''.join(charArr))
                self.wordID[self.vocab[-1].decode('utf-8')] = v
                self.M[v] = struct.unpack('{:d}f'.format(embSize), f.read(4 * embSize))
                lCount += 1
                if lCount % dotCut == 0:
                    sys.stderr.write('.')
                if lCount % nlCut == 0:
                    print >> sys.stderr, '[{:d}]'.format(lCount)

            print >> sys.stderr, "[{:d} items in total loaded]".format(lCount)
            f.close()

if __name__ == '__main__':

    # Open file and read header information
    loader = W2VLoader(sys.argv[1])
    outName = sys.argv[2]
    M = np.memmap('{}.mat.mmap'.format(outName), dtype='float32', mode='w+', shape=(loader.vocSize, loader.embSize))
    M[:] = loader.M[:]
    for w in loader.vocab:
        print w
    
    # Close the file and pickle embedding
    M.flush()
    del M
