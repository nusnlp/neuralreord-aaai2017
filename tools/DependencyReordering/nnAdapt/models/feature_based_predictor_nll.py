"""
This code computes the order of two sibling nodes in a dependency subtree,
where the left and the right siblings are defined on the source dependency tree.
Each node is represented by the continuous vector of its dependency link to its
parent node.

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from mlp import MLP
from logistic_sgd import LogisticRegression
from lookuptable import LookupTable


class FeatureBasedPredictor(object):
    """Multi-Layer Perceptron Class with Lookup Table Input
    """

    def __init__(self, rng, input, feature_size, emb_mat, hidden_sizes=None, activation=T.nnet.relu):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type emb_mat: theano.tensor.TensorType
        :param emb_mat: embedding matrix, must be pre-initialized (random or from w2v)
        """

        # Since we are dealing with a one lookup table layer LR, we need lookup table before LR
        self.lookupTableLayer = LookupTable(
            rng=rng, input=input, emb_mat=emb_mat
        )

        # The projection layer, i.e., an MLP layer, gets as input the output units
        # of the lookup table layer
        emb_size = emb_mat.shape[1]
        if hidden_sizes is None:
            hidden_sizes = [emb_size] # default: 1 hidden layer, same dimension as the input
        self.projectionLayer = MLP(
            rng=rng,
            input=self.lookupTableLayer.output,
            n_in=feature_size * emb_size,
            n_hiddens=hidden_sizes,
            activation=activation
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = self.projectionLayer.L1

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = self.projectionLayer.L2_sqr

        # prediction of the sibling pair order is given by the prediction of the
        # model, computed in the multilayer perceptron projection layer
        self.y_pred = self.projectionLayer.y_pred
        # same holds for the function computing the errors
        self.errors = self.projectionLayer.errors
        # and for the function computing the cross-entropy
        self.cross_entropy = self.projectionLayer.cross_entropy

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.lookupTableLayer.params + self.projectionLayer.params

        # keep track of model input
        self.input = input
        #self.lookupTable = self.lookupTableLayer.embeddings.get_value()[:-1]  # this is erroneous: never update

    def lookupTable(self):  # return current state of the lookup table
        return self.lookupTableLayer.embeddings.get_value()[:-1]

