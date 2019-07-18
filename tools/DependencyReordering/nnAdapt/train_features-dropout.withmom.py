#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: link_embedding_adapt.py
# Christian Hadiwinoto, 2016

"""
This program trains SMT neural reordering model using dependency word pairs (Hadiwinoto et al., 2017)

Reference:
- Christian Hadiwinoto & Hwee Tou Ng. 2017. A neural dependency-based reordering model for statistical machine translation. In AAAI-17.

"""

__docformat__ = 'restructedtext en'

import argparse
import cPickle
import gzip
import sys, os
import time

import math
import numpy

import theano
import theano.tensor as T

from ioWrapper.mmapReader import MemMapReader
from ioWrapper.numpy_to_w2v import save_numpy_w2v
from ioWrapper.numpy_to_w2v import save_numpy_hidden
from ioWrapper.numpy_to_w2v import save_numpy_output
from ioWrapper.w2v_to_numpy import W2VLoader
from models.feature_based_predictor_dropout import FeatureBasedPredictor as Classifier
from algorithms.lr_tuner import LRTuner

def parse_intstr(intstr, delim=','):
    return [int(i) for i in intstr.split(delim)]

def parse_floatstr(fstr, delim=','):
    return [float(i) for i in fstr.split(delim)]

def load_data(trainName, batch_size=1000):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset numpy memory map
    
    :type dataset: batch_size
    :param dataset: the batch size

    '''

    #############
    # LOAD DATA #
    #############
    trainData = MemMapReader(trainName, batch_size)
    return trainData


def train_projections(vocab_file, w2v_file, tnName, devName, output_file,
                      init_learning_rate=0.1, n_epochs=8, update_factor=10, batch_size=128,
                      clip_threshold=0, n_hiddens=None, momentum=0.8, isNesterov=False,
                      dropout_rates=None, n_slack=0, class_weights=None, isFixLR=False, useF=False,
                      dynWeightAfter=-1, weightInertia=0.5, isInitIgnoreOut=False, isInitAccountDropout=False, randomSeed=1234):
    """
    Perform stochastic gradient descent optimization of a log-linear binary model

    :type init_learning_rate: float
    :param init_learning_rate: initial learning rate used (factor for the stochastic
                               gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type n_hiddens: list or None
    :param n_hiddens: predefined number of hidden layer, or None, indicating automatic assignment

    """
    # data set
    trainData = load_data(trainName, batch_size)
    devData = load_data(devName, 1024)

    # compute number of minibatches for training, validation and testing
    n_train_batches = trainData.get_num_batches()
    n_dev_batches = devData.get_num_batches()
    lr_tuner = LRTuner(low=0.01*init_learning_rate, high=10*init_learning_rate, inc=0.01*init_learning_rate)

    # class weights
    uniform_weights = numpy.ones((2,), dtype=theano.config.floatX)  # constant
    class_weights = numpy.asarray(parse_floatstr(class_weights)) if class_weights is not None else uniform_weights  # default is uniform
    print >> sys.stderr, 'Instance weights: positive = %f; negative = %f' % (class_weights[1], class_weights[0])

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print >> sys.stderr, '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.imatrix('x')    # data, presented as word indices and reordering class
    y = T.ivector('y')    # labels, presented as 1D vector of [float] binary labels
    cw = T.fvector('cw')  # class weights, presented as 1D vector of [float] weights

    rng = numpy.random.RandomState(randomSeed)
    
    # load memory-mapped word embedding
    print >> sys.stderr, '... loading word embedding'
    vFile = open(vocab_file, 'r')
    vocab = [l.decode('utf-8').strip() for l in vFile]
    vFile.close()

    fp = numpy.memmap(w2v_file, dtype=theano.config.floatX, mode='r')
    n_voc, n_emb = (len(vocab), fp.shape[0] / len(vocab))
    fp = fp.reshape(n_voc, n_emb)
    emb_mat = numpy.zeros((n_voc, n_emb), dtype=theano.config.floatX)
    emb_mat[:] = fp[:]
    
    print >> sys.stderr, '... defining functions'
    # construct the classifier class
    n_hiddens = n_hiddens if n_hiddens is not None else []
    
    # check if dropout is initialized properly
    if dropout_rates is None: # default is all 0.5
        dropout_rates = [0.5] * (len(n_hiddens)+1)
    else:
        assert len(n_hiddens) + 1 == len(dropout_rates), "ERROR: There are %d hidden layers but there are %d dropout coefficients provided." % (len(n_hiddens), len(dropout_rates))
    classifier = Classifier(rng=rng, input=x, feature_size=trainData.get_feature_size(), emb_mat=emb_mat, hidden_sizes=n_hiddens, dropout_rates=dropout_rates,
        n_slack=n_slack, init_ignore_out=isInitIgnoreOut, init_account_dropout=isInitAccountDropout
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.dropout_cross_entropy(y, cw)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: devData.get_x(index),
            y: devData.get_y(index)
        }
    )
    validate_model_tp = theano.function(
        inputs=[index],
        outputs=classifier.true_positives(y),
        givens={
            x: devData.get_x(index),
            y: devData.get_y(index)
        }
    )
    validate_model_tn = theano.function(
        inputs=[index],
        outputs=classifier.true_negatives(y),
        givens={
            x: devData.get_x(index),
            y: devData.get_y(index)
        }
    )
    validate_model_fp = theano.function(
        inputs=[index],
        outputs=classifier.false_positives(y),
        givens={
            x: devData.get_x(index),
            y: devData.get_y(index)
        }
    )
    validate_model_fn = theano.function(
        inputs=[index],
        outputs=classifier.false_negatives(y),
        givens={
            x: devData.get_x(index),
            y: devData.get_y(index)
        }
    )
    def validate_tp():
        return math.fsum([validate_model_tp(i) for i in xrange(n_dev_batches)])
    def validate_tn():
        return math.fsum([validate_model_tn(i) for i in xrange(n_dev_batches)])
    def validate_fp():
        return math.fsum([validate_model_fp(i) for i in xrange(n_dev_batches)])
    def validate_fn():
        return math.fsum([validate_model_fn(i) for i in xrange(n_dev_batches)])

    def validate_fpos():
        dev_tp = validate_tp()
        dev_fp = validate_fp()
        dev_fn = validate_fn()
        return 2 * dev_tp / (2 * dev_tp + dev_fp + dev_fn)

    def validate_fneg():
        dev_tn = validate_tn()
        dev_fp = validate_fp()
        dev_fn = validate_fn()
        return 2 * dev_tn / (2 * dev_tn + dev_fp + dev_fn)

    def validate_models(): # wrapper function
        if useF:
            dev_tp = math.fsum([validate_model_tp(i) for i in xrange(n_dev_batches)])
            dev_tn = math.fsum([validate_model_tn(i) for i in xrange(n_dev_batches)])
            dev_fp = math.fsum([validate_model_fp(i) for i in xrange(n_dev_batches)])
            dev_fn = math.fsum([validate_model_fn(i) for i in xrange(n_dev_batches)])
            return 1. - (dev_tp / (2. * dev_tp + dev_fp + dev_fn) + dev_tn / (2. * dev_tn + dev_fp + dev_fn) ) # 1 - (F_p + F_n) / 2
        else:
            return math.fsum([validate_model(i) for i in xrange(n_dev_batches)]) / n_dev_batches

    # compute the gradient of cost with respect to theta = (W,b)
    params = classifier.params

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs, with momentum updates
    # Based on http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/nesterov-momentum.php
    lr = T.fscalar()
    updates = []
    if isNesterov:
        for param in params:
            param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))  # a.k.a. velocity
            gparam = T.clip(T.grad(cost, param), -clip_threshold, clip_threshold) if clip_threshold > 0 else T.grad(cost, param)
            updates.append((param_update, momentum * param_update - lr * gparam))
            inc = updates[-1][1]
            inc = momentum * inc - lr * gparam
            updates.append((param, param + inc))  # parameter update
    else:
        for param in params:
            param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))  # a.k.a. velocity
            updates.append((param, param + param_update))  # parameter update
            eval_param = param
            gparam = T.clip(T.grad(cost, eval_param), -clip_threshold, clip_threshold) if clip_threshold > 0 else T.grad(cost, eval_param)
            updates.append((param_update, momentum * param_update - lr * gparam))

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, lr, cw],
        outputs=cost,
        updates=updates,
        givens={
            x: trainData.get_x(index),
            y: trainData.get_y(index),
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print >> sys.stderr, '... training the model'
    # early-stopping parameters
    validation_freq = 10000 # minibatches to go through before checking the network
    verbose_freq = 1000 # minibatches
    update_freq = math.ceil(n_train_batches / float(update_factor))

    start_time = time.clock()  # seconds

    learning_rate = init_learning_rate

    dev_error = validate_models()
    criteria = "1-F1" if useF else "error"
    print >> sys.stderr, 'Initial dev %s %f %%' % (criteria, dev_error * 100.)
    if useF:
        print >> sys.stderr, 'TP = %d; TN = %d; FP = %d; FN = %d' % (validate_tp(), validate_tn(), validate_fp(), validate_fn())
    best_dev_error = dev_error  # initial is the best
    epoch = 0  # Not started yet
    best_epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        print >> sys.stderr, '=== Epoch %i ===' % epoch
        if dynWeightAfter != -1 and epoch > dynWeightAfter: # after N epoch completed
            fneg = validate_fneg()
            fpos = validate_fpos()
            new_weights = numpy.asarray([1.-fneg, 1.-fpos], dtype=theano.config.floatX) * 2. / (2.-fneg-fpos)
            print >> sys.stderr, "Raw weights at epoch %i: positive = %f; negative = %f" % (epoch, new_weights[1], new_weights[0])
            class_weights = weightInertia * uniform_weights + (1-weightInertia) * new_weights
            print >> sys.stderr, "Moderated weights at epoch %i: positive = %f; negative = %f" % (epoch, class_weights[1], class_weights[0])

        minibatch_avg_cost_sum = 0
        epoch_start_time = time.clock()
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, learning_rate, class_weights)
            minibatch_avg_cost_sum += minibatch_avg_cost
            mb_end_time = time.clock()
            if (minibatch_index + 1) % verbose_freq == 0:
                print >> sys.stderr, 'epoch %i, minibatch %i/%i, training cost %f, learning rate %f, %f minibatches/sec.' % (
                    epoch, minibatch_index + 1, n_train_batches, minibatch_avg_cost_sum / (minibatch_index + 1), learning_rate, (minibatch_index + 1) /(mb_end_time - epoch_start_time)
                )
            
            if (minibatch_index + 1) % update_freq == 0:
                dev_error = validate_models()
                if (not isFixLR):
                    learning_rate = lr_tuner.adapt_lr(dev_error, learning_rate)
                print >> sys.stderr, 'epoch %i, minibatch %i/%i, dev %s %f %%' % (epoch, minibatch_index + 1, n_train_batches, criteria, dev_error * 100.)
                if useF:
                    print >> sys.stderr, 'epoch %i, minibatch %i/%i, TP = %d; TN = %d; FP = %d; FN = %d' % (epoch, minibatch_index+1, n_train_batches, validate_tp(), validate_tn(), validate_fp(), validate_fn())
            elif (minibatch_index + 1) % validation_freq == 0:
                dev_error = validate_models()
                print >> sys.stderr, 'epoch %i, minibatch %i/%i, dev %s %f %%' % (epoch, minibatch_index + 1, n_train_batches, criteria, dev_error * 100.)
                if useF:
                    print >> sys.stderr, 'epoch %i, minibatch %i/%i, TP = %d; TN = %d; FP = %d; FN = %d' % (epoch, minibatch_index+1, n_train_batches, validate_tp(), validate_tn(), validate_fp(), validate_fn())
           
        # End of an epoch
        mb_end_time = time.clock()
        print >> sys.stderr, 'epoch %i, minibatch %i/%i, training cost %f, learning rate %f, %f minibatches/sec' % (
            epoch, n_train_batches, n_train_batches, minibatch_avg_cost_sum / n_train_batches, learning_rate, n_train_batches /(mb_end_time - epoch_start_time)
        )
        dev_error = validate_models()
        if (not isFixLR) and n_train_batches % update_freq != 0: # last updating was not done
            learning_rate = lr_tuner.adapt_lr(dev_error, learning_rate)

        isBestStr = ''
        if dev_error < best_dev_error:
            best_dev_error = dev_error
            best_epoch = epoch
            save_numpy_w2v(classifier.lookupTable(), vocab, output_file + '.w2v.best.bin')
            for i in xrange(len(n_hiddens)):
                save_numpy_hidden(classifier.get_W(i),
                    classifier.get_b(i), output_file + '.h{:d}.best.bin'.format(i)) # hidden layer
            save_numpy_output(classifier.get_logre_W(), classifier.get_logre_b(), output_file + '.o.best.bin') # output layer
            isBestStr = ' BEST'
        else:
            save_numpy_w2v(classifier.lookupTable(), vocab, output_file + '.w2v.latest.bin')
            for i in xrange(len(n_hiddens)):
                save_numpy_hidden(classifier.get_W(i),
                    classifier.get_b(i), output_file + '.h{:d}.latest.bin'.format(i)) # hidden layer
            save_numpy_output(classifier.get_logre_W(),
                classifier.get_logre_b(), output_file + '.o.latest.bin') # output layer
        print >> sys.stderr, 'Epoch %i dev %s %f %%%s' % (epoch, criteria, dev_error * 100., isBestStr)
        if useF:
            print >> sys.stderr, 'Epoch %i, TP = %d; TN = %d; FP = %d; FN = %d' % (epoch, validate_tp(), validate_tn(), validate_fp(), validate_fn())

    end_time = time.clock()
    print >> sys.stderr, 'Optimization complete with best validation score of %f %%,' % (best_dev_error * 100.)
    print >> sys.stderr, 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))

if __name__ == '__main__':
    # *** Parameters ***
    pparser = argparse.ArgumentParser()
    pparser.add_argument("-vcb", "--vocab-file", dest="vocab_file_name", required=True, help="Vocabulary (word list) file, sorted by the word index")
    pparser.add_argument("-emb", "--embedding-file", dest="w2v_file_name", required=True, help="Embedding file in Numpy memory-mapped format")
    pparser.add_argument("-tr", "--train-file", dest="train_file_name", required=True, help="Training set file in Numpy memory-mapped format")
    pparser.add_argument("-dev", "--dev-file", dest="dev_file_name", required=True, help="Development set file in Numpy memory-mapped format")
    pparser.add_argument("-out", "--output-file-prefix", dest="output_file_name", required=True, help="File prefix for model output")
    pparser.add_argument("-E", "--num-epochs", dest="num_epochs", default=8, type=int, help="Number of epochs hyperparameter")
    pparser.add_argument("-mb", "--batch-size", dest="batch_size", default=128, type=int, help="Number of instances per minibatchhyperparameter")
    pparser.add_argument("-lr", "--learning-rate", dest="alpha", type=float, required=True, help="Initial learning rate")
    pparser.add_argument("-H", "--hidden-units", dest="n_hiddens", help="A comma seperated list for the number of units in each hidden layer")
    pparser.add_argument("--clip-threshold", dest="clip_threshold", default=0, type=float, help="If threshold > 0, clips gradients to [-threshold, +threshold]. Default: 0 (disabled)")
    pparser.add_argument("--momentum", dest="momentum", default=0.8, type=float, help="Momentum for gradient descent")
    pparser.add_argument("--nesterov", dest="is_nesterov", action='store_true', help="Enable Nesterov modified momentum")
    pparser.add_argument("--update-factor", dest="update_factor", type=int, default=1, help="Update learnign rate this number of time per epoch")
    pparser.add_argument("-do", "--dropout", dest="dropout_rates", help="A comma separated list for dropout rates in each input and hidden layer")
    pparser.add_argument("--slack-dimension", dest="n_slack", type=int, default=0, help="Number of dimensions after lookup table set to 0, useful for initialization")
    pparser.add_argument("--class_weights", dest="class_weights", help="Weights for negative and positive classes separated by comma")
    pparser.add_argument("--fix_lr", dest="is_fix_lr", action='store_true', help="Use fixed learning rate throughout training")
    pparser.add_argument("--use_f", dest="is_use_f", action='store_true', help="Use F-score to validate model (error rate = 1-F)")
    pparser.add_argument("-dw", "--dynamic-weight-after", dest="dyn_weight_after", type=int, default=-1, help="Start tuning class weights after N epochs. Default not used (-1)")
    pparser.add_argument("-wi", "--weight-inertia", dest="weight_inertia", type=float, default=0.5, help="For dynamic class weighting: keep uniform weight stronger by this factor")
    pparser.add_argument("--init-ignore-output", dest="init_ignore_output", action='store_true', help="Ignore output size for initialization")
    pparser.add_argument("--init-account-dropout", dest="init_account_dropout", action='store_true', help="Account dropout coefficient size for initialization (initialize with larger weight range)")
    pparser.add_argument("-rs", "--random-seed", dest="random_seed", type=int, default=1234, help="Random seed to initialize pseudo-random")

    params = pparser.parse_args()
    params.cwd = os.getcwd()
    # word (link string) embedding
    vocab_file_name = params.vocab_file_name
    w2v_file_name = params.w2v_file_name
    
    # dataset in memory mapped file
    trainName = params.train_file_name
    devName = params.dev_file_name
    output_file_name = params.output_file_name
    lr = params.alpha
    
    if params.n_hiddens:
        n_hiddens = parse_intstr(params.n_hiddens)
    else:
        n_hiddens = None

    train_projections(vocab_file_name, w2v_file_name, trainName, devName, output_file_name, init_learning_rate=lr, n_hiddens=n_hiddens,
        clip_threshold=params.clip_threshold, momentum=params.momentum, isNesterov=params.is_nesterov, n_epochs=params.num_epochs, batch_size=params.batch_size,
        dropout_rates=parse_floatstr(params.dropout_rates) if params.dropout_rates else None, update_factor=params.update_factor, n_slack=params.n_slack, class_weights=params.class_weights,
        isFixLR=params.is_fix_lr, useF=params.is_use_f, dynWeightAfter=params.dyn_weight_after, weightInertia=params.weight_inertia, isInitIgnoreOut=params.init_ignore_output, 
        isInitAccountDropout=params.init_account_dropout,randomSeed=params.random_seed)
