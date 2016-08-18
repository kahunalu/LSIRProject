# Python Libraries
import cPickle
import math
from random import shuffle
from random import randint

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano import printing
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

final_prediction = []

#Covnet
class covnet:
	def __init__(self, layers, mini_batch_size):

		# Splits
		self.training_data		= []
		self.test_data			= []
		self.validation_data	= []

		# Network Architecture
		self.layers				= layers
		self.mini_batch_size	= mini_batch_size		
		self.params 			= [param for layer in self.layers for param in layer.params]
		self.x					= T.matrix("x")
		self.y					= T.ivector("y")
		init_layer				= self.layers[0]
		self.final_layer		= self.layers[-1]
		
		init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
		
		for j in xrange(1, len(self.layers)):
			prev_layer, layer	= self.layers[j-1], self.layers[j]

			layer.set_inpt(
				prev_layer.output, prev_layer.output_dropout, self.mini_batch_size
			)

		self.output = self.layers[-1].output
		self.output_dropout = self.layers[-1].output_dropout

	#Create random splits
	def _split_sets(self):
		dataset = np.load("/home/mclaren1/seng/LSIRProject/classifiers/basicCovnet/used/ksh_shuffle_data.dat")
		labels = np.load("/home/mclaren1/seng/LSIRProject/classifiers/basicCovnet/used/ksh_shuffle_labels.dat")

		imagenet_dataset = np.load("/home/mclaren1/seng/LSIRProject/classifiers/basicCovnet/imagenet/ksh_imagenet_data.dat")
                imagenet_labels = np.load("/home/mclaren1/seng/LSIRProject/classifiers/basicCovnet/imagenet/ksh_imagenet_labels.dat")

		# Create splits
		length = len(dataset)
		imagenet_length = len(imagenet_dataset)

		__training_data, __test_data, __validation_data = [[],[]],[[],[]],[[],[]]

		__training_data[0] = dataset[0:int(length*0.9)]
		__training_data[1] = labels[0:int(length*0.9)]

		__test_data[0] = imagenet_dataset[int(imagenet_length*0.8):(imagenet_length*0.9)]
		__test_data[1] = imagenet_labels[int(imagenet_length*0.8):(imagenet_length*0.9)]

		__validation_data[0] = dataset[int(length*0.9):length]
		__validation_data[1] = labels[int(length*0.9):length]

		def shared(data):
			shared_x = theano.shared(
				np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
			shared_y = theano.shared(
				np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
			return shared_x, T.cast(shared_y, "int32")

		#Return splits
		return shared(__training_data), shared(__test_data), shared(__validation_data)

	def create_splits(self, filename="./dataset.dat"):

		self.training_data, self.test_data, self.validation_data = self._split_sets()
		
	#Train network using mini-batch gradient descent
	def SGD(self, epochs, mini_batch_size, eta, lmbda=0.0):

		# Split sets into x and y aka. images and categories
		training_x, training_y 		= self.training_data
		validation_x, validation_y 	= self.validation_data
		test_x, test_y 				= self.test_data

		# compute number of minibatches for training, validation and testing
		num_training_batches	= size(training_x)/mini_batch_size
		num_validation_batches 	= size(validation_x)/mini_batch_size
		num_test_batches		= size(test_x)/mini_batch_size

		# define the (regularized) cost function, symbolic gradients, and updates
		l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
		cost = self.layers[-1].cost(self)+\
			   0.5*lmbda*l2_norm_squared/num_training_batches
		grads = T.grad(cost, self.params)
		updates = [(param, param-eta*grad)
				   for param, grad in zip(self.params, grads)]

		# define functions to train a mini-batch, and to compute the
		# accuracy in validation and test mini-batches.
		i = T.lscalar() # mini-batch index
		train_mb = theano.function(
			[i], cost, updates=updates,
			givens={
				self.x:
				training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
				self.y:
				training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
			})
		validate_mb_accuracy = theano.function(
			[i], self.layers[-1].accuracy(self.y),
			givens={
				self.x:
				validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
				self.y:
				validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
			})
		test_mb_accuracy = theano.function(
			[i], self.layers[-1].accuracy(self.y),
			givens={
				self.x:
				test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
				self.y:
				test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
			})
		self.test_mb_predictions = theano.function(
			[i], self.layers[-1].y_out,
			givens={
				self.x:
				test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
			})

		final_prediction = []

		# Do the actual training
		best_validation_accuracy = 0.0
		for epoch in xrange(epochs):
			final_prediction = []
			for minibatch_index in xrange(num_training_batches):
				iteration = num_training_batches*epoch+minibatch_index
				if iteration % 1000 == 0:
					print("Training mini-batch number {0}".format(iteration))
				cost_ij = train_mb(minibatch_index)
				if (iteration+1) % num_training_batches == 0:

					validation_accuracy_array = [validate_mb_accuracy(j) for j in xrange(num_validation_batches)]
					validation_accuracy = np.mean([pair[0] for pair in validation_accuracy_array])

					print("Epoch {0}: validation accuracy {1:.2%}".format(
						epoch, validation_accuracy))
					if validation_accuracy >= best_validation_accuracy:
						print("This is the best validation accuracy to date.")
						best_validation_accuracy = validation_accuracy
						best_iteration = iteration
						if self.test_data:

							test_accuracy_array = [test_mb_accuracy(j) for j in xrange(num_test_batches)]


							test_accuracy = np.mean([pair[0] for pair in test_accuracy_array])
							np.save(str(test_accuracy)+"used_predictions_list"+str(randint(1,100)), np.asarray([pair[1] for pair in test_accuracy_array]))
							np.save(str(test_accuracy)+"used_test_list"+str(randint(1,100)), np.asarray([pair[2] for pair in test_accuracy_array]))

							print('The corresponding test accuracy is {0:.2%}'.format(
								test_accuracy))

		print "Prediction\n"
		print accuracy_array

		print("Finished training network.")
		print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
			best_validation_accuracy, best_iteration))
		print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types
class ConvPoolLayer(object):
	"""Used to create a combination of a convolutional and a max-pooling
	layer.  A more sophisticated implementation would separate the
	two, but for our purposes we'll always use them together, and it
	simplifies the code, so it makes sense to combine them.

	"""

	def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
				 activation_fn=sigmoid):
		"""`filter_shape` is a tuple of length 4, whose entries are the number
		of filters, the number of input feature maps, the filter height, and the
		filter width.

		`image_shape` is a tuple of length 4, whose entries are the
		mini-batch size, the number of input feature maps, the image
		height, and the image width.

		`poolsize` is a tuple of length 2, whose entries are the y and
		x pooling sizes.

		"""

		self.filter_shape 	= filter_shape
		self.image_shape 	= image_shape
		self.poolsize 		= poolsize
		self.activation_fn	= activation_fn
		
		# initialize weights and biases
		n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
		self.w = theano.shared(
			np.asarray(
				np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
				dtype=theano.config.floatX),
			borrow=True)
		self.b = theano.shared(
			np.asarray(
				np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
				dtype=theano.config.floatX),
			borrow=True)
		self.params = [self.w, self.b]

	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		self.inpt = inpt.reshape(self.image_shape)
		conv_out = conv.conv2d(
			input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
			image_shape=self.image_shape)
		pooled_out = downsample.max_pool_2d(
			input=conv_out, ds=self.poolsize, ignore_border=True)
		self.output = self.activation_fn(
			pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

	def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
		self.n_in = n_in
		self.n_out = n_out
		self.activation_fn = activation_fn
		self.p_dropout = p_dropout
		# Initialize weights and biases
		self.w = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
				dtype=theano.config.floatX),
			name='w', borrow=True)
		self.b = theano.shared(
			np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
					   dtype=theano.config.floatX),
			name='b', borrow=True)
		self.params = [self.w, self.b]

	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		self.inpt = inpt.reshape((mini_batch_size, self.n_in))
		self.output = self.activation_fn(
			(1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
		self.y_out = T.argmax(self.output, axis=1)
		self.inpt_dropout = dropout_layer(
			inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
		self.output_dropout = self.activation_fn(
			T.dot(self.inpt_dropout, self.w) + self.b)

	def accuracy(self, y):
		"Return the accuracy for the mini-batch."
		return T.mean(T.eq(y, self.y_out)), self.y_out

class SoftmaxLayer(object):

	def __init__(self, n_in, n_out, p_dropout=0.0):
		self.n_in = n_in
		self.n_out = n_out
		self.p_dropout = p_dropout
		# Initialize weights and biases
		self.w = theano.shared(
			np.zeros((n_in, n_out), dtype=theano.config.floatX),
			name='w', borrow=True)
		self.b = theano.shared(
			np.zeros((n_out,), dtype=theano.config.floatX),
			name='b', borrow=True)
		self.params = [self.w, self.b]

	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		self.inpt = inpt.reshape((mini_batch_size, self.n_in))
		self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
		self.y_out = T.argmax(self.output, axis=1)
		self.inpt_dropout = dropout_layer(
			inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
		self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

	def cost(self, net):
		"Return the log-likelihood cost."
		return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

	def accuracy(self, y):
		print "Return the accuracy for the mini-batch."
		return T.mean(T.eq(y, self.y_out)), self.y_out, y

#### Miscellanea
def size(data):
	return data.get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
	srng = shared_randomstreams.RandomStreams(
		np.random.RandomState(0).randint(999999))
	mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
	return layer*T.cast(mask, theano.config.floatX)

mini_batch_size = 100

covnet = covnet([
	ConvPoolLayer(image_shape=(mini_batch_size, 1, 224, 224), 
				  filter_shape=(48, 1, 10, 10),
				  poolsize=(5, 5)),
	FullyConnectedLayer(n_in=48*43*43, n_out=100),
	SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

print "Starting Covnet"

print "Creating Splits"
covnet.create_splits()

print "Start training"
covnet.SGD(60, mini_batch_size, 0.1)  
