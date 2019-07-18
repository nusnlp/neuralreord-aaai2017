from __future__ import division
import theano.tensor as T
import theano
import numpy

class LRTuner:
	def __init__(self, low, high, inc):
		self.low = low
		self.high = high
		self.inc = inc
		self.prev_error = numpy.inf
	
	def adapt_lr(self, curr_error, curr_lr):
		if curr_error >= self.prev_error:
			lr = max(curr_lr / 2, self.low)
		else:
			lr = min(curr_lr + self.inc, self.high)
		self.prev_error = curr_error
		return lr
