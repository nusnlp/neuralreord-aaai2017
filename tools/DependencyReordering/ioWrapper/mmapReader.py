import numpy as np
import theano
import theano.tensor as T
import math as M
import sys
import os

class MemMapReader():
    
    #### Constructor
    
    def __init__(self, dataset_path, batch_size=500, fix_batch_num=False):
        
        print >> sys.stderr, "Initializing dataset from: " + os.path.abspath(dataset_path)
        
        # Reading parameters from the mmap file
        fp = np.memmap(dataset_path, dtype='int32', mode='r')
        self.feature_size = fp[1] - 1
        self.num_samples = fp.shape[0] / (self.feature_size + 1) - 1
        print >> sys.stderr, "Expecting %d samples, %d features, %d data" % (self.num_samples, self.feature_size, fp.shape[0])
        fp = fp.reshape((self.num_samples + 1, self.feature_size + 1))

        # Setting minibatch size and number of mini batches
        self.batch_size = None
        self.num_batches = None
        if fix_batch_num:
            self.num_batches = batch_size
            self.batch_size = int(M.ceil(self.num_samples / float(self.num_batches)))
        else:
            self.batch_size = batch_size
            self.num_batches = int(M.ceil(self.num_samples / float(self.batch_size)))
        
        # Reading the matrix of samples (label is recast to float for cross-entropy computation
        x = fp[1:,:-1]          # Reading the context indices
        y = fp[1:,-1]           # Reading the output word index
        self.shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
        self.shared_y = T.cast(theano.shared(y, borrow=True), 'int32')
        
        print >> sys.stderr, '  #samples: %d, ngram size: %d, batch size: %d, #batches: %d' % (self.num_samples, self.feature_size, self.batch_size, self.num_batches)
    
    #### Accessors
    
    def get_x(self, index):
        return self.shared_x[index * self.batch_size : (index+1) * self.batch_size]
    
    def get_y(self, index):
        return self.shared_y[index * self.batch_size : (index+1) * self.batch_size]
    
    #### INFO
    
    def get_num_samples(self):
        return self.num_samples
    
    def get_num_batches(self):
        return self.num_batches
    
    def get_feature_size(self):
        return self.feature_size
