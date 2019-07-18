import sys
import theano
import theano.tensor as T
import numpy as np
from w2v_to_numpy import W2VLoader

class LookupTable():
    
    def __init__(self, rng, input, emb_mat):
        (vocab_size, emb_dim) = emb_mat.shape
        print >> sys.stderr, "Lookup Table layer, #words: %d, #dims: %d" % (vocab_size, emb_dim)

        self.input = input
        self.embeddings = theano.shared(np.append(emb_mat, np.zeros((1, emb_dim), dtype=theano.config.floatX), axis=0), borrow=True)
        self.params = [self.embeddings]

        mask = np.append(np.ones((vocab_size,), dtype=theano.config.floatX), np.asarray([0.], dtype=theano.config.floatX))[:, None]
        masked_embeddings = self.embeddings * mask  # to facilitate for blank entries

        self.output = masked_embeddings[input].reshape((input.shape[0], emb_dim * input.shape[1]))
