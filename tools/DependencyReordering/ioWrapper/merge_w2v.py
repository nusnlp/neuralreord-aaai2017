#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: merge_w2v.py
# Merge (concatenate) word2vec file
import sys
import struct
import numpy as np
from w2v_to_numpy import W2VLoader

if __name__ == '__main__':
    emb1Name = sys.argv[1]
    emb2Name = sys.argv[2]
    outName = sys.argv[3]

    # Open numpized word2vec file
    emb1 = W2VLoader(emb1Name)
    vcbIdx1 = {}
    for i, v in enumerate(emb1.vocab):
        vcbIdx1[v] = i
    emb2 = W2VLoader(emb2Name)
    vcbIdx2 = {}
    for i, v in enumerate(emb2.vocab):
        vcbIdx2[v] = i

    print >> sys.stderr, "Intersecting two vocabulary sets ..."
    vcbSet1 = set(emb1.vocab)
    vcbSet2 = set(emb2.vocab)
    vcbJoint = list(vcbSet1.intersection(vcbSet2))
    embSizeJoint = emb1.embSize + emb2.embSize
    print >> sys.stderr, "Got %d items." % len(vcbJoint)
    
    f = open(outName, 'wb')
    print >> sys.stderr, "Writing binary word2vec file"
    lCount = 0
    dotCut = 5000
    dpl = 100
    nlCut = dpl * dotCut
    print >> f, "{:d} {:d}".format(len(vcbJoint), emb1.embSize + emb2.embSize)
    for i, w in enumerate(vcbJoint):
        f.write('{} '.format(w))
        #mj = np.concatenate(emb1.M[vcbIdx1[w]], emb2.M[vcbIdx2[w]])
        f.write(struct.pack('{:d}f'.format(emb1.embSize), *tuple(emb1.M[vcbIdx1[w]])))
        f.write(struct.pack('{:d}f'.format(emb2.embSize), *tuple(emb2.M[vcbIdx2[w]])))
        f.write('\n')
        lCount += 1
        if lCount % dotCut == 0:
            sys.stderr.write('.')
        if lCount % nlCut == 0:
            print >> sys.stderr, '[%d]' % lCount

    f.close()
