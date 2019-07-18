"""
This code is adapted from Deep Learning tutorial introducing multilayer perceptron
using Theano.
Added Dropout from https://github.com/mdenil/dropout/blob/master/mlp.py

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from logistic_sgd_beta_zeros import LogisticRegression
from mlp import HiddenLayer


def _dropout_from_layer(rng, layer, p):
    """p is the probability of droping a unit
    """
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1.-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, activation, dropout_rate, W=None, b=None, n_slack=0, dilate_factor=1.):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, activation=activation, n_slack=n_slack, dilate_factor=dilate_factor)
        print >> sys.stderr, 'Slack size =', str(n_slack)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate) if dropout_rate > 0 else self.output

# start-snippet-2
class DropoutMLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hiddens, dropout_rates, activation=None, n_slack=0, init_ignore_out=False, init_account_dropout=False):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: list of int
        :param n_hidden: number of hidden units

        :type n_slack: int
        :param n_slack: number of slack input units, which are always 0 on average

        :type init_ignore_out: boolean
        :param init_ignore_out: ignore output size when initializing, default False

        :type init_account_dropout: boolean
        :param init_account_dropout: account dropout coefficient when initializing, i.e, dilate dropout weights by inverse dropout coefficient

        """
        self.params = []
        self.W = []
        self.b = []

        self.W_actual = []
        self.b_actual = []

        # keep track of model input
        self.input = input

        # Multiple hidden layers
        print >> sys.stderr, dropout_rates
        last_layer_out = self.input
        last_layer_dropout = _dropout_from_layer(rng, self.input, p=dropout_rates[0])
        last_layer_size = n_in

        slacks = numpy.append(numpy.asarray([n_slack], dtype='int32'), numpy.zeros((len(n_hiddens)-1,), dtype='int32')) if len(n_hiddens) > 0 else None
        for i in range(0, len(n_hiddens)):
            # dropped-out path: for training
            dropoutLayer = DropoutHiddenLayer(rng=rng,
                input=last_layer_dropout, activation=activation,
                n_in=last_layer_size, n_out=n_hiddens[i],
                dropout_rate=dropout_rates[i+1],
                n_slack=slacks[i] + (n_hiddens[i] if init_ignore_out else 0),
                dilate_factor = (1./(1. - dropout_rates[i]) if init_account_dropout else 1.)
            )
            last_layer_dropout = dropoutLayer.output

            self.params += dropoutLayer.params
            self.W += [dropoutLayer.W]
            self.b += [dropoutLayer.b]
            
            # original (untouched) path: for testing
            hiddenLayer = HiddenLayer(rng=rng,
                input=last_layer_out, activation=activation,
                n_in=last_layer_size, n_out=n_hiddens[i],
                W=dropoutLayer.W * (1. - dropout_rates[i]),
                b=dropoutLayer.b,
                n_slack=slacks[i]
            )
            last_layer_out = hiddenLayer.output
            last_layer_size = n_hiddens[i]

            self.W_actual += [hiddenLayer.W]
            self.b_actual += [hiddenLayer.b]

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        # Dropped-out path: for training
        self.dropoutLogRegressionLayer = LogisticRegression(
            rng=rng, input=last_layer_dropout,
            n_in=(n_hiddens[-1] if len(n_hiddens) > 0 else n_in)
        )
        self.params += self.dropoutLogRegressionLayer.params

        # original (untouched) path: for testing
        self.logRegressionLayer = LogisticRegression(
            rng=rng, input=last_layer_out,
            n_in=(n_hiddens[-1] if len(n_hiddens) > 0 else n_in),
            W=self.dropoutLogRegressionLayer.W * (1. - dropout_rates[-1]),
            b=self.dropoutLogRegressionLayer.b
        )

        # prediction of the MLP is given by the prediction of the output of the
        # model, computed in the logistic regression layer
        self.dropout_errors = self.dropoutLogRegressionLayer.errors
        self.dropout_cross_entropy = self.dropoutLogRegressionLayer.cross_entropy

        self.y_pred = self.logRegressionLayer.y_pred
        self.errors = self.logRegressionLayer.errors
        self.cross_entropy = self.logRegressionLayer.cross_entropy

        self.true_positives = self.logRegressionLayer.true_positives
        self.true_negatives = self.logRegressionLayer.true_negatives
        self.false_positives = self.logRegressionLayer.false_positives
        self.false_negatives = self.logRegressionLayer.false_negatives

