"""
VAE + Pixel CNN
Ishaan Gulrajani
"""

"""
Modified by Kundan Kumar
Usage: THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=.95' python experiments/pixel_only/mnist.py -L 10 -F 5 
"""
import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import time
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

import argparse

parser = argparse.ArgumentParser(description='Generating images pixel by pixel')
parser.add_argument('-L','--num_pixel_cnn_layer', required=True, type=int, help='Number of layers to use in pixelCNN')
parser.add_argument('-F','--pixel_filter_size', required=True, type=int, help='filter_size to use in pixelCNN')

args = parser.parse_args()
# theano.config.dnn.conv.algo_fwd = 'time_on_shape_change'
# theano.config.dnn.conv.algo_bwd_filter = 'time_on_shape_change'
# theano.config.dnn.conv.algo_bwd_data = 'time_on_shape_change'

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

OUT_DIR = '/Tmp/kumarkun/mnist_pixel_only_new' + "/layer_{}_fs_{}".format(args.num_pixel_cnn_layer, args.pixel_filter_size)

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)
    print "Created directory {}".format(OUT_DIR)

DIM_1 = 32
DIM_2 = 32
DIM_3 = 64
DIM_4 = 64
DIM_PIX = 32
PIXEL_CNN_FILTER_SIZE = args.pixel_filter_size
PIXEL_CNN_LAYERS = args.num_pixel_cnn_layer

LATENT_DIM = 32
ALPHA_ITERS = 10000
VANILLA = False
LR = 1e-3

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TEST_BATCH_SIZE = 100
TIMES = ('iters', 500, 2000*500, 1000, 200*500, 2*ALPHA_ITERS)

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

def next_stacks_only_h(X_h, inp_dim, name, 
                global_conditioning = None, 
                filter_size = 3, 
                hstack = 'hstack', 
                residual = True
            ):
    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.hstack', 
            input_dim= inp_dim, 
            output_dim= 2*DIM_PIX, 
            filter_size= (filter_size,filter_size), 
            inputs= X_h, 
            mask_type=(hstack, N_CHANNELS)
        )

    X_h_next = PixCNNGate(X_h_next)

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.h2h', 
            input_dim=DIM_PIX, 
            output_dim=DIM_PIX, 
            filter_size=(1,1), 
            inputs= X_h_next
            )

    if residual == True:
        X_h_next = X_h_next + X_h

    return X_h_next


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

def Decoder_pixelCNN_single_stack(images):

    X_h = next_stacks_only_h(
                images, N_CHANNELS, "Dec.PixInput", 
                filter_size = 7, 
                hstack = "hstack_a", residual = False
                )

    for i in xrange(PIXEL_CNN_LAYERS):
        X_h = next_stacks_only_h(X_h, DIM_PIX, "Dec.Pix"+str(i+1), filter_size = PIXEL_CNN_FILTER_SIZE)
    
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


def get_every_layer_functions():
    functions = []
    img = T.tensor4('img')
    X_v, X_h = next_stacks(
                img, img, N_CHANNELS, "Dec.PixInput", 
                filter_size = 7, 
                hstack = "hstack_a", residual = False
                )
    print "Compiling layer 0 ......."
    functions.append(theano.function([img], [X_v, X_h]))

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v_inp = T.tensor4('X_v_inp')
        X_h_inp = T.tensor4('X_h_inp')
        X_v, X_h = next_stacks(X_v_inp, X_h_inp, DIM_PIX, "Dec.Pix"+str(i+1), filter_size = PIXEL_CNN_FILTER_SIZE)
        print "Compiling layer {} .......".format(i+1)
        functions.append(theano.function([X_v_inp, X_h_inp], [X_v, X_h]))

    X_h = T.tensor4('X_h')
    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=X_h)
    output = PixCNNGate(output)
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = PixCNNGate(output)
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)
    output = T.nnet.sigmoid(output)
    print "Compiling output function :)"
    functions.append(theano.function([X_h], output))


    return functions

def get_every_layer_functions_only_h():
    functions = []
    img = T.tensor4('img')
    X_h = next_stacks_only_h(
                img, N_CHANNELS, "Dec.PixInput", 
                filter_size = 7, 
                hstack = "hstack_a", residual = False
                )
    print "Compiling layer 0 ......."
    functions.append(theano.function([img], X_h))

    for i in xrange(PIXEL_CNN_LAYERS):
        X_h_inp = T.tensor4('X_h_inp')
        X_h = next_stacks_only_h(X_h_inp, DIM_PIX, "Dec.Pix"+str(i+1), filter_size = PIXEL_CNN_FILTER_SIZE)
        print "Compiling layer {} .......".format(i+1)
        functions.append(theano.function([X_h_inp],  X_h))

    X_h = T.tensor4('X_h')
    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=X_h)
    output = PixCNNGate(output)
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = PixCNNGate(output)
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)
    output = T.nnet.sigmoid(output)
    print "Compiling output function :)"
    functions.append(theano.function([X_h], output))


    return functions

# def auto_regress(shape):
#     images = T.zeros(shape)
Decoder = Decoder_pixelCNN
# Decoder = Decoder_pixelCNN_single_stack

total_iters = T.iscalar('total_iters')
images = T.tensor4('images') # shape: (batch size, n channels, height, width)

# Theano bug: NaNs unless I pass 2D tensors to binary_crossentropy
reconst_cost = T.nnet.binary_crossentropy(
    T.nnet.sigmoid(
        Decoder(images).reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
    ),
    images.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
).sum(axis=1).mean()

cost = reconst_cost

sample_fn = theano.function(
    [images],
    T.nnet.sigmoid(Decoder(images))
)

eval_fn = theano.function(
    [images],
    cost
)

train_data, dev_data, test_data = lib.mnist_binarized.load(
    BATCH_SIZE, 
    TEST_BATCH_SIZE
)

def get_receptive_area(h,w, receptive_field, i, j):
    if i < receptive_field:
        i_min = 0
        i_end = 2*receptive_field + 1
        i_res = i
    elif i >= (h - receptive_field):
        i_end = h
        i_min = h - (2*receptive_field + 1)
        i_res = i - i_min
    else:
        i_min = i - receptive_field
        i_end = i + receptive_field + 1
        i_res = i - i_min

    if j < receptive_field:
        j_min = 0
        j_end = 2*receptive_field + 1
        j_res = j
    elif j >= (w - receptive_field):
        j_end = w
        j_min = w - (2*receptive_field + 1)
        j_res = j - j_min
    else:
        j_min = j - receptive_field
        j_end = j + receptive_field + 1
        j_res = j - j_min

    return i_min, i_end, i_res, j_min, j_end, j_res

def generate_with_only_receptive_field(samples):
    h, w =  HEIGHT, WIDTH
    receptive_field = 3 + ((PIXEL_CNN_FILTER_SIZE/2)*PIXEL_CNN_LAYERS)

    t0 = time.time()
    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                j_min, j_end, j_res, k_min, k_end, k_res = get_receptive_area(h,w, receptive_field, j,k)
                res = binarize(sample_fn(samples[:,:,j_min:j_end, k_min:k_end]))
                samples[:, i, j, k] = res[:, i, j_res, k_res]

    t1 = time.time()
    print("Time taken is {:.4f}s".format(t1 - t0))

    return samples

def get_dependence_field(i,j, X, num_channels, batch_size,  filter_size):
    h, w = HEIGHT, WIDTH

    region = np.zeros((batch_size, num_channels, filter_size, filter_size)).astype(theano.config.floatX)

    Xi_beg = (i - (filter_size//2))
    Xi_end = filter_size - (filter_size//2) + i

    ri_beg = 0
    ri_end = filter_size

    Xj_beg = (j - (filter_size//2))
    Xj_end = filter_size - (filter_size//2) + j

    rj_beg = 0
    rj_end = filter_size

    if Xi_beg < 0:
        ri_beg = (filter_size//2) - i
        Xi_beg = 0
    elif Xi_end > h:
        Xi_end = h
        ri_end = Xi_end - Xi_beg

    if Xj_beg < 0:
        rj_beg = (filter_size//2) - j
        Xj_beg = 0
    elif Xj_end > w:
        Xj_end = w
        rj_end = Xj_end - Xj_beg

    region[:, :, ri_beg:ri_end, rj_beg:rj_end] = X[:, :, Xi_beg:Xi_end, Xj_beg:Xj_end]
        
    return region

#########################
##### Test Region predictors
# samples = np.arange(28*28).reshape((1,1,28,28)).astype(theano.config.floatX)

# print get_dependence_field(0,0,samples, 1, 1, 5)[0,0]
# print get_dependence_field(27,27,samples, 1, 1, 5)[0,0]
# print get_dependence_field(0,27,samples, 1, 1, 5)[0,0]
# print get_dependence_field(27,0,samples, 1, 1, 5)[0,0]
# exit()
####
#########################

print "creating functions..."
# layer_functions = get_every_layer_functions()
layer_functions = get_every_layer_functions_only_h()

assert(len(layer_functions) == (PIXEL_CNN_LAYERS + 2))

def faster_generation(functions):
    assert(N_CHANNELS == 1), "Current version supports only 1 input_channel i.e. dependency amongst the channels cannoit be modelled"
    
    h, w = HEIGHT, WIDTH

    samples = np.zeros(
        (100, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=theano.config.floatX
    )
    binarized_samples = samples.copy()

    temp_Xv_s = []
    temp_Xh_s = []
    
    for i in xrange(PIXEL_CNN_LAYERS + 1):
        temp_Xv_s.append(
                    np.zeros((100, DIM_PIX, HEIGHT, WIDTH), dtype=theano.config.floatX)
                )
        temp_Xh_s.append(
                    np.zeros((100, DIM_PIX, HEIGHT, WIDTH), dtype=theano.config.floatX)
                )

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                samples_slice = get_dependence_field(j, k, binarized_samples, N_CHANNELS, 100, 9)
                X_v_next, X_h_next = functions[0](samples_slice)

                temp_Xh_s[0][:,:,j,k] = X_h_next[:,:,4,4]
                temp_Xv_s[0][:,:,j,k] = X_v_next[:,:,4,4]

                for l in range(PIXEL_CNN_LAYERS):
                    curr_fun = functions[l+1]
                    X_h_temp = get_dependence_field(j, k, temp_Xh_s[l], DIM_PIX, 100, PIXEL_CNN_FILTER_SIZE+2)
                    X_v_temp = get_dependence_field(j, k, temp_Xv_s[l], DIM_PIX, 100, PIXEL_CNN_FILTER_SIZE+2)
                    
                    X_v_next, X_h_next = curr_fun(X_v_temp, X_h_temp)

                    temp_Xh_s[l+1][:,:,j,k] = X_h_next[:,:,((PIXEL_CNN_FILTER_SIZE//2) + 1),((PIXEL_CNN_FILTER_SIZE//2) + 1)]
                    temp_Xv_s[l+1][:,:,j,k] = X_v_next[:,:,((PIXEL_CNN_FILTER_SIZE//2) + 1),((PIXEL_CNN_FILTER_SIZE//2) + 1)]


                sampler = functions[-1]

                output =  sampler(temp_Xh_s[-1][:,:,j:j+1,k:k+1])

                binarized_samples[:,i,j,k] = binarize(output)[:,i,0,0]
                samples[:, i, j, k] = output[:, i, 0, 0]

    return samples
    
def faster_generation_only_h(functions):
    assert(N_CHANNELS == 1), "Current version supports only 1 input_channel i.e. dependency amongst the channels cannoit be modelled"
    
    h, w = HEIGHT, WIDTH

    samples = np.zeros(
        (100, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=theano.config.floatX
    )
    binarized_samples = samples.copy()

    temp_Xh_s = []
    
    for i in xrange(PIXEL_CNN_LAYERS + 1):
        temp_Xh_s.append(
                    np.zeros((100, DIM_PIX, HEIGHT, WIDTH), dtype=theano.config.floatX)
                )

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                samples_slice = get_dependence_field(j, k, binarized_samples, N_CHANNELS, 100, 9)
                X_h_next = functions[0](samples_slice)

                temp_Xh_s[0][:,:,j,k] = X_h_next[:,:,4,4]

                for l in range(PIXEL_CNN_LAYERS):
                    curr_fun = functions[l+1]
                    X_h_temp = get_dependence_field(j, k, temp_Xh_s[l], DIM_PIX, 100, PIXEL_CNN_FILTER_SIZE+2)
                    X_h_next = curr_fun(X_h_temp)

                    temp_Xh_s[l+1][:,:,j,k] = X_h_next[:,:,((PIXEL_CNN_FILTER_SIZE//2) + 1),((PIXEL_CNN_FILTER_SIZE//2) + 1)]
                    

                sampler = functions[-1]

                output =  sampler(temp_Xh_s[-1][:,:,j:j+1,k:k+1])

                binarized_samples[:,i,j,k] = binarize(output)[:,i,0,0]
                samples[:, i, j, k] = output[:, i, 0, 0]

    return samples
    

def binarize(images):
        """
        Stochastically binarize values in [0, 1] by treating them as p-values of
        a Bernoulli distribution.
        """
        return (
            np.random.uniform(size=images.shape) < images
        ).astype(theano.config.floatX)

def generate_and_save_samples(tag):

    costs = []
    # for (images,) in test_data():
    #     costs.append(eval_fn(images))
    # print "test cost: {}".format(np.mean(costs))
    lib.save_params(os.path.join(OUT_DIR, tag + "_params.pkl"))

    def save_images(images, filename):
        """images.shape: (batch, n channels, height, width)"""
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))

        image = scipy.misc.toimage(images, cmin=0.0, cmax=1.0)
        image.save('{}/{}_{}.jpg'.format(OUT_DIR, filename, tag))

    samples = np.zeros(
        (100, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=theano.config.floatX
    )

    next_sample = samples.copy()

    t0 = time.time()
    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                samples_p_value = sample_fn(next_sample)
                next_sample[:,i,j,k] = binarize(samples_p_value)[:,i,j,k]
                samples[:, i, j, k] = samples_p_value[:, i, j, k]

    t1 = time.time()
    save_images(samples, 'samples')
    print("Time taken with slowest generation is {:.4f}s".format(t1 - t0))

    # t0 = time.time()
    # samples =  faster_generation_only_h(layer_functions)
    # t1 = time.time()
    # print("Time taken with faster generation is {:.4f}s".format(t1 - t0))

    # save_images(samples, 'samples_faster_generation')

    # samples = generate_with_only_receptive_field(samples)
    # save_images(samples, 'samples_receptive_field')
# lib.load_params(os.path.join(OUT_DIR, 'iters15500_time30678.3401728_params.pkl'))
generate_and_save_samples("initial_samples")
# exit()
print("Training..")

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