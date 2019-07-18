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


from dropout_mlp_nll import DropoutMLP as MLP
from lookuptable import LookupTable


class FeatureBasedPredictor(object):
    """Multi-Layer Perceptron Class with Lookup Table Input
    """

    def __init__(self, rng, input, feature_size, emb_mat, hidden_sizes=None, activation=T.nnet.relu, dropout_rates=None, n_slack=0):
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
        self.dropout_rates = dropout_rates
        self.lookupTableLayer = LookupTable(
            rng=rng, input=input, emb_mat=emb_mat
        )

        # The projection layer, i.e., an MLP layer, gets as input the output units
        # of the lookup table layer
        emb_size = emb_mat.shape[1]
        if hidden_sizes is None:
            hidden_sizes = [] # default: no hidden layer
        if dropout_rates is None:
            dropout_rates = [0.5] * (len(hidden_sizes)+1)
        self.projectionLayer = MLP(
            rng=rng,
            input=self.lookupTableLayer.output,
            n_in=feature_size * emb_size,
            n_hiddens=hidden_sizes,
            n_out=2,  # swap or no-swap
            dropout_rates=dropout_rates,
            activation=activation,
            n_slack=n_slack
        )

        # prediction of the sibling pair order is given by the prediction of the
        # model, computed in the multilayer perceptron projection layer
        self.y_pred = self.projectionLayer.y_pred
        self.errors = self.projectionLayer.errors
        self.negative_log_likelihood = self.projectionLayer.negative_log_likelihood

        self.dropout_errors = self.projectionLayer.dropout_errors
        self.dropout_negative_log_likelihood = self.projectionLayer.dropout_negative_log_likelihood

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.lookupTableLayer.params + self.projectionLayer.params

        # keep track of model input
        self.input = input
    
    # Access to actual (non-dropped) parameters
    def lookupTable(self):  # return current state of the lookup table
        return self.lookupTableLayer.embeddings.get_value()[:-1]

    def get_W(self, i):
        return self.projectionLayer.W[i].get_value() * (1. - self.dropout_rates[i])

    def get_b(self, i):
        return self.projectionLayer.b[i].get_value()

    def get_logre_W(self):
        return self.projectionLayer.dropoutLogRegressionLayer.W.get_value() * (1. - self.dropout_rates[-1])

    def get_logre_b(self):
        return self.projectionLayer.dropoutLogRegressionLayer.b.get_value()


