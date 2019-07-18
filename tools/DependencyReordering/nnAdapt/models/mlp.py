"""
This code is adapted from Deep Learning tutorial introducing multilayer perceptron
using Theano.

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, n_slack=0, dilate_factor=1.):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type n_slack: int
        :param n_slack: number of slack dimension (average always 0 in one item)
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        if W is None:
            if activation == T.tanh:
                print >> sys.stderr, 'Activation: tanh'
                high = numpy.sqrt(6. / (n_in - n_slack + n_out)) * dilate_factor
            elif activation == T.nnet.sigmoid:
                print >> sys.stderr, 'Activation: sigmoid'
                high = 4 * numpy.sqrt(6. / (n_in - n_slack + n_out)) * dilate_factor
            elif activation == T.nnet.relu:
                print >> sys.stderr, 'Activation: Relu'  # He et al (2015) initialization, modified from Glorot & Bengio (2010)
                high = numpy.sqrt(2. / (n_in - n_slack + n_out)) * dilate_factor
            else:
                print >> sys.stderr, 'Activation: None'
                high = 0.01 * dilate_factor
            
            W_values = numpy.asarray(
                rng.uniform(
                    low=-high, high=high,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hiddens, activation=None, init_ignore_out=False):
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

        """
        self.params = []
        self.L1 = 0
        self.L2_sqr = 0
        self.W = []
        self.b = []

        # keep track of model input
        self.input = input

        # Multiple hidden layers
        last_layer_out = self.input
        last_layer_size = n_in
        for i in range(0, len(n_hiddens)):
            hiddenLayer = HiddenLayer(
                rng=rng, input=last_layer_out,
                n_in=last_layer_size, n_out=n_hiddens[i],
                n_slack=(n_hiddens[i] if init_ignore_out else 0),
                activation=activation
            )
            last_layer_out = hiddenLayer.output
            last_layer_size = n_hiddens[i]
            self.params += hiddenLayer.params
            
            self.L1 = self.L1 + abs(hiddenLayer.W).sum()
            self.L2_sqr = self.L2_sqr + (hiddenLayer.W ** 2).sum()
            self.W += [hiddenLayer.W]
            self.b += [hiddenLayer.b]

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            rng=rng,
            input=last_layer_out,
            n_in=n_hiddens[-1]
        )
        self.params += self.logRegressionLayer.params
        self.L1 = self.L1 + self.logRegressionLayer.L1
        self.L2_sqr = self.L1 + self.logRegressionLayer.L2_sqr

        # prediction of the MLP is given by the prediction of the output of the
        # model, computed in the logistic regression layer
        self.y_pred = self.logRegressionLayer.y_pred
        # same holds for the function computing the errors
        self.errors = self.logRegressionLayer.errors
        # and for the function computing the cross-entropy
        self.cross_entropy = self.logRegressionLayer.cross_entropy

