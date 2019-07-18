#!/usr/bin/python
"""
This code is adapted from Deep Learning tutorial introducing logistic regression
using Theano and stochastic gradient descent.

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Binary Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, rng, input, n_in):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=rng.uniform(
                low=-numpy.sqrt(6. / n_in),
                high=numpy.sqrt(6. / n_in),
                size=(n_in, )
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=0., name='b',
            borrow=True
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.W ** 2).sum()

        # symbolic expression for computing the probability of positive case (p_1)
        # Where:
        # h_1 is the linear component for the logit
        # W is a vector of separation hyperplane for positive case
        # x is a matrix where row-j  represents input training sample-j
        # b is the free parameter of the positive case
        h_1 = T.dot(input, self.W) + self.b
        self.p_1 = 1 / (1 + T.exp(-h_1))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = (self.p_1 > 0.5)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def cross_entropy(self, y, class_weights=None):
        """Return the mean of the cross-entropy
        assuming L2 regularization

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vecvtor that gives for each example the correct label

        """
        if class_weights is None:
            class_weights = (1., 1.)
        return -T.mean(class_weights[1] * T.cast(y, theano.config.floatX) * T.log(self.p_1) + class_weights[0] * (1-T.cast(y, theano.config.floatX)) * T.log(1-self.p_1))
        
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

