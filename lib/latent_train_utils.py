import theano
import theano.tensor as T

import lasagne

from sklearn.svm import SVC
import numpy

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

def train_on_batch(Encoder, train_data, test_data):
	"TODO: To be implemented"

def train_svm(Encoder, train_data, evaluate_data , dimension = 16, **kwargs):
	Y_train = []
	Z_train = numpy.zeros((50000, dimension), dtype = theano.config.floatX)
	i = 0
	for (images, targets) in train_data():
		new_examples = len(targets)
		Y_train += list(targets.astype(numpy.int32))
		Z_train[i: i + new_examples] = Encoder(images)

		i += new_examples

	assert (i == 50000)

	print "Training SVM....."
	clf = SVC()
	clf.fit(Z_train, Y_train)

	print "Training accuracy : {}".format(clf.score(Z_train, Y_train))

	correct_classification = 0
	for (images, targets) in evaluate_data():
		z = Encoder(images)
		predictions = clf.predict(z)

		correct_classification += numpy.sum((targets == predictions))

	evaluation_accuracy = correct_classification/10000.0

	print "Evaluation accuracy : {}".format(evaluation_accuracy)










