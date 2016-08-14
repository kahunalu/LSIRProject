# Python Libraries
import math

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
def ReLU(z):
	return T.maximum(0.0, z)

from theano.tensor.nnet import hard_sigmoid

#Covnet
class covnet:
	def __init__(self, layers, mini_batch_size):

		# Splits
		self.training_data		= []
		self.test_data			= []

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

	def create_splits(self, label_folder, image_folder, ext, iteration, test):
		imageset = np.load(image_folder+"images"+str(iteration)+ext)
		labelset = np.load(label_folder+"labels"+str(iteration)+ext)

		print "DONE DONE DONE DONE "
		print len(imageset)
		print len(labelset)

		# Create splits
		length = len(imageset)

		def shared(data):
			shared_x = theano.shared(
				np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
			shared_y = theano.shared(
				np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
			return shared_x, T.cast(shared_y, "int32")

		self.training_data, self.test_data = [[],[]],[[],[]]

		#if test iteration create test set
		if test:
			print "Creating Testing Split"
			for i in range(0, length):
				self.test_data[0].append(imageset[i])
				self.test_data[1].append(labelset[i])
			np.save('dnnnet_imagenet_test_file', self.test_data)
			print self.test_data
			self.test_data = shared(self.test_data)
		else:
			print "Creating Training  Split"
			for i in range(0, length):
				self.training_data[0].append(imageset[i])
				self.training_data[1].append(labelset[i])
			self.training_data = shared(self.training_data)

	#Train network using mini-batch gradient descent
	def SGD(self, epochs, mini_batch_size, eta, lmbda=0.0, test=False):

		# define functions to train a mini-batch, and to compute the
		# accuracy in validation and test mini-batches.
		i = T.lscalar() # mini-batch index

		#If the test flag is shown test the covnet with the current test set
		if test:
			test_x, test_y 				= self.test_data

			num_test_batches			= size(test_x)/mini_batch_size

			test_mb_accuracy = theano.function(
				[i], self.layers[-1].accuracy(),
				givens={
					self.x: 
						test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
				}
			)

			test_accuracy = [test_mb_accuracy(j) for j in xrange(num_test_batches)]

			np.save("dnn_used_test", np.asarray(test_accuracy))

		#Else train the network
		else:
			training_x, training_y 		= self.training_data
			
			num_training_batches	= size(training_x)/mini_batch_size

			# define the (regularized) cost function, symbolic gradients, and updates
			l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
			cost = self.layers[-1].cost(self)+\
				   0.5*lmbda*l2_norm_squared/num_training_batches
			grads = T.grad(cost, self.params)
			updates = [(param, param-eta*grad)
					   for param, grad in zip(self.params, grads)]
			
			#Train theano function
			train_mb = theano.function(
				[i], cost, updates=updates,
				givens={
					self.x:
					training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
					self.y:
					training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
				}
			)

			for epoch in xrange(epochs):
				print "Running epoch #"+str(epoch)+" of "+ str(epochs)
				for minibatch_index in xrange(num_training_batches):
					cost_ij = train_mb(minibatch_index)

			print("Finished training network.")


#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=ReLU):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
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

    def __init__(self, n_in, n_out, activation_fn=ReLU, p_dropout=0.0):
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
        return T.mean(T.eq(y, self.y_out))

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
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self):
        "Return the accuracy for the mini-batch."
        return self.y_out


#### Miscellanea
def size(data):
	return data.get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
	srng = shared_randomstreams.RandomStreams(
		np.random.RandomState(0).randint(999999))
	mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
	return layer*T.cast(mask, theano.config.floatX)


'''
DEFINE THE CONV NEURAL NETWORK
'''

print "Begin used DNN"

mini_batch_size = 100

covnet = covnet([
	ConvPoolLayer(image_shape=(mini_batch_size, 1, 460, 614),
					filter_shape=(48, 1, 7, 7),
					poolsize=(2, 2)),
	ConvPoolLayer(image_shape=(mini_batch_size, 48, 227, 304),
					filter_shape=(128, 48, 8, 9),
					poolsize=(2, 2)),
	ConvPoolLayer(image_shape=(mini_batch_size, 128, 110, 148),
					filter_shape=(192, 128, 5, 5),
					poolsize=(2, 2)),
	FullyConnectedLayer(n_in=(192*53*72), n_out=100),
	SoftmaxLayer(n_in=100, n_out=10)], 
	mini_batch_size)


print "Start training Covnet"
for i in range(0,6):
	covnet.create_splits(
		label_folder="/home/mclaren1/seng/LSIRProject/data/used/label_folder/",
		image_folder="/home/mclaren1/seng/LSIRProject/data/used/bw_data_folder/",
		ext=".dat",
		iteration=i,
		test=False
	)

	covnet.SGD(1, mini_batch_size, 0.1, test=False)

#Test the accuracy of the covnet
np.save('dnnnet_imagenet_params_'+str(i), covnet.params)

#Create Test Split for imagenet
covnet.create_splits(
	label_folder="/home/mclaren1/seng/LSIRProject/data/imagenet/label_folder/",
	image_folder="/home/mclaren1/seng/LSIRProject/data/imagenet/bw_data_folder/",
	ext=".dat",
	iteration=6,
	test=True
)

#Test the network
covnet.SGD(0, mini_batch_size, 0.1, test=True)
