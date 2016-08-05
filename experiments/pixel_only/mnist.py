"""
VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist_binarized
import lib.ops.mlp
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne

import functools

# theano.config.dnn.conv.algo_fwd = 'time_on_shape_change'
# theano.config.dnn.conv.algo_bwd_filter = 'time_on_shape_change'
# theano.config.dnn.conv.algo_bwd_data = 'time_on_shape_change'

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

OUT_DIR = '/Tmp/kumarkun/mnist_pixel_only' + "/layer_12"

if not os.path.isdir(OUT_DIR):
    os.system('mkdir {}'.format(OUT_DIR))

DIM_1 = 32
DIM_2 = 32
DIM_3 = 64
DIM_4 = 64
DIM_PIX = 32
PIXEL_CNN_FILTER_SIZE = 5
PIXEL_CNN_LAYERS = 12

LATENT_DIM = 64
ALPHA_ITERS = 10000
VANILLA = False
LR = 1e-3

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TEST_BATCH_SIZE = 100
TIMES = ('iters', 1000, 2000*500, 1000, 200*500, 2*ALPHA_ITERS)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def PixCNNGate(x):
    a = x[:,::2]
    b = x[:,1::2]
    return T.tanh(a) * T.nnet.sigmoid(b)

def PixCNN_condGate(x, z, dim, name = ""):
    a = x[:,::2]
    b = x[:,1::2]

    Z_to_tanh = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
    Z_to_sigmoid = lib.ops.linear.Linear(name+".sigmoid", input_dim=LATENT_DIM, output_dim=dim, inputs=z)

    a = a + Z_to_tanh[:,:, None, None]
    b = b + Z_to_sigmoid[:,:,None, None]
    return T.tanh(a) * T.nnet.sigmoid(b)

def next_stacks(X_v, X_h, inp_dim, name, 
                global_conditioning = None, 
                filter_size = 3, 
                hstack = 'hstack', 
                residual = True
            ):
    zero_pad = T.zeros((X_v.shape[0], X_v.shape[1], 1, X_v.shape[3]))

    X_v_padded = T.concatenate([zero_pad, X_v], axis = 2)

    X_v_next = lib.ops.conv2d.Conv2D(
            name + ".vstack", 
            input_dim=inp_dim, 
            output_dim=2*DIM_PIX, 
            filter_size=filter_size, 
            inputs=X_v_padded, 
            mask_type=('vstack', N_CHANNELS)
        )

    X_v_next_gated = PixCNNGate(X_v_next)

    X_v2h = lib.ops.conv2d.Conv2D(
            name + ".v2h", 
            input_dim=2*DIM_PIX, 
            output_dim=2*DIM_PIX, 
            filter_size=(1,1), 
            inputs=X_v_next[:,:,:-1,:]
        )

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.hstack', 
            input_dim= inp_dim, 
            output_dim= 2*DIM_PIX, 
            filter_size= (1,filter_size), 
            inputs= X_h, 
            mask_type=(hstack, N_CHANNELS)
        )

    X_h_next = PixCNNGate(X_h_next + X_v2h)

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.h2h', 
            input_dim=DIM_PIX, 
            output_dim=DIM_PIX, 
            filter_size=(1,1), 
            inputs= X_h_next
            )

    if residual == True:
        X_h_next = X_h_next + X_h

    return X_v_next_gated[:, :, 1:, :], X_h_next

def next_stacks_gated(X_v, X_h, inp_dim, name, global_conditioning = None,
                                             filter_size = 3, hstack = 'hstack', residual = True):
    zero_pad = T.zeros((X_v.shape[0], X_v.shape[1], 1, X_v.shape[3]))

    X_v_padded = T.concatenate([zero_pad, X_v], axis = 2)

    X_v_next = lib.ops.conv2d.Conv2D(
            name + ".vstack", 
            input_dim=inp_dim, 
            output_dim=2*DIM_PIX, 
            filter_size=filter_size, 
            inputs=X_v_padded, 
            mask_type=('vstack', N_CHANNELS)
        )
    X_v_next_gated = PixCNN_condGate(X_v_next, global_conditioning, DIM_PIX,
                                     name = name + ".vstack.conditional")

    X_v2h = lib.ops.conv2d.Conv2D(
            name + ".v2h", 
            input_dim=2*DIM_PIX, 
            output_dim=2*DIM_PIX, 
            filter_size=(1,1), 
            inputs=X_v_next[:,:,:-1,:]
        )
    

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.hstack', 
            input_dim= inp_dim, 
            output_dim= 2*DIM_PIX, 
            filter_size= (1,filter_size), 
            inputs= X_h, 
            mask_type=(hstack, N_CHANNELS)
        )

    X_h_next = PixCNN_condGate(X_h_next + X_v2h, global_conditioning, DIM_PIX, name = name + ".hstack.conditional")

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.h2h', 
            input_dim=DIM_PIX, 
            output_dim=DIM_PIX, 
            filter_size=(1,1), 
            inputs= X_h_next
            )

    if residual:
        X_h_next = X_h_next + X_h

    return X_v_next_gated[:, :, 1:, :], X_h_next


def Decoder_pixelCNN(images):

    X_v, X_h = next_stacks(
                images, images, N_CHANNELS, "Dec.PixInput", 
                filter_size = 7, 
                hstack = "hstack_a", residual = False
                )

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h, DIM_PIX, "Dec.Pix"+str(i+1), filter_size = PIXEL_CNN_FILTER_SIZE)
    
    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=X_h)
    output = PixCNNGate(output)
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = PixCNNGate(output)
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

total_iters = T.iscalar('total_iters')
images = T.tensor4('images') # shape: (batch size, n channels, height, width)

# Theano bug: NaNs unless I pass 2D tensors to binary_crossentropy
reconst_cost = T.nnet.binary_crossentropy(
    T.nnet.sigmoid(
        Decoder_pixelCNN(images).reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
    ),
    images.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
).mean(axis=0).sum()

cost = reconst_cost

sample_fn = theano.function(
    [images],
    T.nnet.sigmoid(Decoder_pixelCNN(images))
)

eval_fn = theano.function(
    [images],
    cost
)

train_data, dev_data, test_data = lib.mnist_binarized.load(
    BATCH_SIZE, 
    TEST_BATCH_SIZE
)

def generate_and_save_samples(tag):

    costs = []
    for (images,) in test_data():
        costs.append(eval_fn(images))
    print "test cost: {}".format(np.mean(costs))

    def save_images(images, filename):
        """images.shape: (batch, n channels, height, width)"""
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))

        image = scipy.misc.toimage(images, cmin=0.0, cmax=1.0)
        image.save('{}/{}_{}.jpg'.format(OUT_DIR, filename, tag))

    def binarize(images):
        """
        Stochastically binarize values in [0, 1] by treating them as p-values of
        a Bernoulli distribution.
        """
        return (
            np.random.uniform(size=images.shape) < images
        ).astype(theano.config.floatX)


    samples = np.zeros(
        (100, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=theano.config.floatX
    )

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                next_sample = binarize(sample_fn(samples))
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, 'samples')

lib.train_loop.train_loop(
    inputs=[images],
    inject_total_iters=False,
    cost=cost,
    prints=[
        ('cost', cost), 
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    test_data=dev_data,
    callback=generate_and_save_samples,
    times=TIMES
)