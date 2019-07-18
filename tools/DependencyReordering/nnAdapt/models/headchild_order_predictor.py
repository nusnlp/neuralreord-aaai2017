"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression
from lookuptable import LookupTable


class HeadChildOrderPredictor(object):
    """Logistic Regression Class with Lookup Table Input

    """

    #def __init__(self, rng, input, n_in, voc_size, emb_size, emb_path=None):
    def __init__(self, rng, input, feature_size, emb_mat):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type voc_size: int
        :param voc_size: vocabulary size (# of words)

        :type emb_size: int
        :param emb_size: embedding vector size

        """

        # Since we are dealing with a one lookup table layer LR, we need lookup table before LR
        self.lookupTableLayer = LookupTable(
            rng=rng, input=input, emb_mat=emb_mat
        )
        emb_size = emb_mat.get_value().shape[1]

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            rng=rng,
            input=self.lookupTableLayer.output,
            n_in=feature_size * emb_size
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()

        # prediction of the head-child pair order is given by the prediction of the
        # model, computed in the logistic regression layer
        self.y_pred = self.logRegressionLayer.y_pred
        # same holds for the function computing the errors
        self.errors = self.logRegressionLayer.errors
        # and for the function computing the cross-entropy
        self.cross_entropy = self.logRegressionLayer.cross_entropy

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.lookupTableLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

