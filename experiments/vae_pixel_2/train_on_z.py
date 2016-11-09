import theano
import theano.tensor as T

import lasagne

def create_mlp(input_dim, hidden_dim, output_dim):
	inp = T.matrix('input')
	Y = T.ivector('labels')

	l_in = lasagne.layers.InputLayer((None, input_dim))
	l_hidden = lasagne.layers.DenseLayer(l_in, num_units=hidden_dim, nonlinearity=T.nnet.tanh)
	l_out = lasagne.layers.DenseLayer(l_hidden, num_units=output_dim, nonlinearity=T.nnet.softmax)

	output = lasagne.layers.get_output(l_out)

	output_class = T.argmax(output, axis=1)

	loss = T.nnet.categorical_crossentropy(output, Y)

	params = lasagne.layers.get_all_params(l_out)

	grads = T.grad(loss, wrt=params, disconnected_inputs='warn')
	grads = [T.clip(g, floatX(-1.), floatX(1.)) for g in grads]



	updates = lasagne.updates.adam(grads, params, learning_rate = 1e-3)

	train_fn = theano.function([X,Y], loss, updates = updates )
	valid_fn = theano.function([X,Y], loss)
	predict_fn = theano.function([X], output_class)

	return train_fn, valid_fn, predict_fn

def train_on_batch(Encoder, data):
	"TODO: To be implemented"


