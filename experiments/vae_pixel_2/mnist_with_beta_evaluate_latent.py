"""
Modified by Kundan Kumar

Usage: THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=.95' python experiments/vae_pixel_2/mnist_with_beta_evaluate_latent.py -L 2 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16 -beta 1.

This code operates on dynamic binarization.
"""

import os, sys
sys.path.append(os.getcwd())

import time

import argparse

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist_stochastic_binarized
import lib.ops.mlp
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu

import lib.latent_train_utils

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne
import pickle

import functools


parser = argparse.ArgumentParser(description='Generating images pixel by pixel')
parser.add_argument('-L','--num_pixel_cnn_layer', required=True, type=int, help='Number of layers to use in pixelCNN')
parser.add_argument('-algo', '--decoder_algorithm', required = True, help="One of 'cond_z_bias', 'upsample_z_no_conv', 'upsample_z_conv', 'upsample_z_conv_tied' 'vae_only'" )
parser.add_argument('-enc', '--encoder', required = False, default='simple', help="Encoder: 'complecated' or 'simple' " )
parser.add_argument('-dpx', '--dim_pix', required = False, default=32, type = int )
parser.add_argument('-fs', '--filter_size', required = False, default=5, type = int )
parser.add_argument('-ldim', '--latent_dim', required = False, default=64, type = int )
parser.add_argument('-ait', '--alpha_iters', required = False, default=10000, type = int )
parser.add_argument('-beta', '--beta', required = False, default=1., type = lib.floatX )
parser.add_argument('-file_to_load', '--file_to_load', required = True, type = str )

args = parser.parse_args()


assert args.decoder_algorithm in ['condRMB','cond_z_bias', 'cond_z_bias_skip', 'upsample_z_no_conv', 'upsample_z_conv', 'upsample_z_conv_tied', 'vae_only', 'traditional', 'traditional_exact' ]

print args

# theano.config.dnn.conv.algo_fwd = 'time_on_shape_change'
# theano.config.dnn.conv.algo_bwd_filter = 'time_on_shape_change'
# theano.config.dnn.conv.algo_bwd_data = 'time_on_shape_change'

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

def floatX(num):
    if theano.config.floatX == 'float32':
        return np.float32(num)
    else:
        raise Exception("{} type not supported".format(theano.config.floatX))


DIM_1 = 32
DIM_2 = 32
DIM_3 = 64
DIM_4 = 64
DIM_PIX = args.dim_pix
PIXEL_CNN_FILTER_SIZE = args.filter_size
PIXEL_CNN_LAYERS = args.num_pixel_cnn_layer

LATENT_DIM = args.latent_dim
ALPHA_ITERS = args.alpha_iters
VANILLA = False
LR = 1e-3
BETA = args.beta

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TEST_BATCH_SIZE = 100
TIMES = ('iters', 500, 500*400, 500, 400*500, 2*ALPHA_ITERS)

# lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)


T.nnet.elu = lambda x: T.switch(x >= floatX(0.), x, T.exp(x) - floatX(1.))



def PixCNNGate(x):
    a = x[:,::2]
    b = x[:,1::2]
    return T.tanh(a) * T.nnet.sigmoid(b)

def PixCNN_condGate(x, z, dim,  activation= 'tanh', name = ""):
    a = x[:,::2]
    b = x[:,1::2]

    Z_to_tanh = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
    Z_to_sigmoid = lib.ops.linear.Linear(name+".sigmoid", input_dim=LATENT_DIM, output_dim=dim, inputs=z)

    a = a + Z_to_tanh[:,:, None, None]
    b = b + Z_to_sigmoid[:,:,None, None]

    if activation == 'tanh':
        return T.tanh(a) * T.nnet.sigmoid(b)
    else:
        return T.nnet.elu(a) * T.nnet.sigmoid(b)

def pixelCNN_old_block(x, inp_dim, name, residual = False):
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(name + ".1x1_init", input_dim=inp_dim, output_dim=DIM_PIX, filter_size=1, inputs=x))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(name + ".3x3_masked", input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(name + ".1x1_final", input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))

    if residual:
        return output + x
    else:
        return output


# def PixCNN_condGate_ELU(x, z, dim, name = ""):
#     a = x[:,::2]
#     b = x[:,1::2]

#     Z_to_tanh = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
#     Z_to_sigmoid = lib.ops.linear.Linear(name+".sigmoid", input_dim=LATENT_DIM, output_dim=dim, inputs=z)

#     a = a + Z_to_tanh[:,:, None, None]
#     b = b + Z_to_sigmoid[:,:,None, None]
#     return T.nnet.elu(a) * T.nnet.sigmoid(b)

def PixCNN_ELU_no_gate(x, z, dim, name = ""):
    Z_to_elu = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
    return T.nnet.elu(a + Z_to_elu[:, :, None, None])

def PixCNN_condNoGate(x, z, dim, name = ""):
    Z_to_tanh = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
    return T.tanh(a + Z_to_tanh[:, :, None, None])

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


def next_stacks_gated_skip(X_v, X_h, inp_dim, name, global_conditioning = None,
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

    X_h_next_ = lib.ops.conv2d.Conv2D(
            name + '.h2h',
            input_dim=DIM_PIX,
            output_dim= 2*DIM_PIX,
            filter_size=(1,1),
            inputs= X_h_next
            )

    if residual:
        X_h_next = X_h_next_[:,::2,:,:] + X_h
    else:
    	X_h_next = X_h_next_[:,::2,:,:]

    return X_v_next_gated[:, :, 1:, :], X_h_next, X_h_next_[:,1::2,:,:]


def MU(name, x, dim, cond = None, mask_type = 'b', filter_size=3):

    x_transformed = lib.ops.conv2d.Conv2D(
            name + '.x_trans_conv',
            input_dim= dim,
            output_dim= 4*dim,
            filter_size= filter_size,
            inputs= x,
            mask_type=(mask_type, N_CHANNELS)
        )

    if cond is not None:
        cond_transform = lib.ops.linear.Linear(name+".cond_transform", input_dim=LATENT_DIM, output_dim=4*dim, inputs=cond)
        x_transformed = x_transformed + cond_transform[:,:, None, None]

    g1 = x_transformed[:,::4]
    g2 = x_transformed[:,1::4]
    g3 = x_transformed[:,2::4]
    g4 = x_transformed[:,3::4]

    out = T.tanh( (T.tanh(g4)*T.nnet.sigmoid(g3)) + (T.nnet.sigmoid(g2)*x) )*T.nnet.sigmoid(g1)

    return out

def condRMB(name, x, dim, cond = None, filter_size = 3):
    x1 = lib.ops.conv2d.Conv2D(
            name + '.x1',
            input_dim=dim,
            output_dim=dim,
            filter_size=(1,1),
            inputs= x
            )

    x2 = MU(name + '.x2', x1, dim, cond = cond, filter_size= filter_size)

    x3 = MU(name + '.x3', x2, dim, cond = cond, filter_size= filter_size)

    x4 = lib.ops.conv2d.Conv2D(
            name + '.x4',
            input_dim=dim,
            output_dim=dim,
            filter_size=(1,1),
            inputs= x3
        )

    return x + x4





def Encoder(inputs):

    output = inputs

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.2', input_dim=DIM_1,      output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

    # Pad from 7x7 to 8x8
    padded = T.zeros((output.shape[0], output.shape[1], 8, 8), dtype='float32')
    output = T.inc_subtensor(padded[:,:,:7,:7], output)

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.6', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.7', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.8', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = output.reshape((output.shape[0], -1))
    output = lib.ops.linear.Linear('Enc.Out', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM, inputs=output)
    return output[:, ::2], output[:, 1::2]

def Encoder_with_elu(inputs):

    output = inputs

    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.2', input_dim=DIM_1,      output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

    # Pad from 7x7 to 8x8
    padded = T.zeros((output.shape[0], output.shape[1], 8, 8), dtype='float32')
    output = T.inc_subtensor(padded[:,:,:7,:7], output)

    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.6', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.7', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.elu(lib.ops.conv2d.Conv2D('Enc.8', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = output.reshape((output.shape[0], -1))
    output = lib.ops.linear.Linear('Enc.9', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM, inputs=output)

    output = T.nnet.elu(output)
    output = lib.ops.linear.Linear('Enc.10', input_dim=2*LATENT_DIM, output_dim=2*LATENT_DIM, inputs=output)

    return output[:, ::2], output[:, 1::2]

def Encoder_complecated(inputs):

    output = inputs

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.2', input_dim=DIM_1,      output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(T.signal.pool.pool_2d(lib.ops.conv2d.Conv2D('Enc.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output), (2,2)))
    # output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

    # Pad from 7x7 to 8x8
    padded = T.zeros((output.shape[0], output.shape[1], 8, 8), dtype='float32')
    output = T.inc_subtensor(padded[:,:,:7,:7], output)

    output = T.nnet.relu(T.signal.pool.pool_2d(lib.ops.conv2d.Conv2D('Enc.5', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output), (2,2)))
    # output = T.signal.pool.pool_2d(lib.ops.conv2d.Conv2D('Enc.6', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.7', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.8', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = output.reshape((output.shape[0], -1))
    output = lib.ops.linear.Linear('Enc.Out', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM, inputs=output)
    return output[:, ::2], output[:, 1::2]

def Decoder(latents, images):

    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    masked_images = T.nnet.relu(lib.ops.conv2d.Conv2D(
        'Dec.PixInp',
        input_dim=N_CHANNELS,
        output_dim=DIM_1,
        filter_size=7,
        inputs=images,
        mask_type=('a', N_CHANNELS),
        he_init=False
    ))

    output = T.concatenate([masked_images, output], axis=1)

    for i in xrange(PIXEL_CNN_LAYERS):
        inp_dim = (DIM_1 + DIM_PIX if i==0 else DIM_PIX)

        # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.Pix'+str(i), input_dim=inp_dim, output_dim=2*DIM_1, filter_size=PIXEL_CNN_FILTER_SIZE, inputs=output, mask_type=('b', N_CHANNELS)))
        output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.Pix'+str(i), input_dim=inp_dim, output_dim=DIM_PIX, filter_size=PIXEL_CNN_FILTER_SIZE, inputs=output, mask_type=('b', N_CHANNELS)))
        # skip_outputs.append(output)
        if i > 0:
            output = output + prev_out
        prev_out = output

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_traditional(latents, images):

    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    masked_images = T.nnet.relu(lib.ops.conv2d.Conv2D(
        'Dec.PixInp',
        input_dim=N_CHANNELS,
        output_dim=DIM_1,
        filter_size=7,
        inputs=images,
        mask_type=('a', N_CHANNELS),
        he_init=False
    ))

    output = T.concatenate([masked_images, output], axis=1)

    for i in xrange(PIXEL_CNN_LAYERS):
        inp_dim = (DIM_1 + DIM_PIX if i==0 else DIM_PIX)

        # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.Pix'+str(i), input_dim=inp_dim, output_dim=2*DIM_1, filter_size=PIXEL_CNN_FILTER_SIZE, inputs=output, mask_type=('b', N_CHANNELS)))
        output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.Pix'+str(i), input_dim=inp_dim, output_dim=DIM_PIX, filter_size=PIXEL_CNN_FILTER_SIZE, inputs=output, mask_type=('b', N_CHANNELS)))
        # skip_outputs.append(output)
        if i > 0:
            output = output + prev_out
        prev_out = output

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output


def Decoder_traditional_exact(latents, images):

    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    masked_images = T.nnet.relu(lib.ops.conv2d.Conv2D(
        'Dec.PixInp',
        input_dim=N_CHANNELS,
        output_dim=DIM_1,
        filter_size=7,
        inputs=images,
        mask_type=('a', N_CHANNELS),
        he_init=False
    ))

    output = T.concatenate([masked_images, output], axis=1)

    for i in xrange(PIXEL_CNN_LAYERS):
        inp_dim = (DIM_1 + DIM_PIX if i==0 else DIM_PIX)

        # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.Pix'+str(i), input_dim=inp_dim, output_dim=2*DIM_1, filter_size=PIXEL_CNN_FILTER_SIZE, inputs=output, mask_type=('b', N_CHANNELS)))
        output = pixelCNN_old_block(output, inp_dim, 'Dec.Pix'+str(i), residual = (i != 0))
        # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_only_vae(latents, iamges):
    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=DIM_PIX, filter_size=1, inputs=output))
    skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    skip_outputs.append(output)

    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output


def Decoder_no_blind(latents, images):
    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    images_with_latent = T.concatenate([images, output], axis=1)

    X_v, X_h = next_stacks(images_with_latent, images_with_latent, N_CHANNELS + DIM_1, "Dec.PixInput", filter_size = 7, hstack = "hstack_a", residual = False)

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h,  DIM_PIX, "Dec.Pix"+str(i+1), filter_size = 3)


    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_1, filter_size=1, inputs=X_h))
    # output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=X_h))
    # skip_outputs.append(output)

    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    # output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_1, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_no_blind_tied_pixelCNN_weights(latents, images):
    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    images_with_latent = T.concatenate([images, output], axis=1)

    X_v, X_h = next_stacks(images_with_latent, images_with_latent, N_CHANNELS + DIM_1, "Dec.PixInput", filter_size = 7, hstack = "hstack_a", residual = False)

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h,  DIM_PIX, "Dec.Pix_tied", filter_size = 3)


    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_1, filter_size=1, inputs=X_h))
    # output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=X_h))
    # skip_outputs.append(output)

    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    # output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_1, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_no_blind_vary_up_sampling(latents, images):
    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    # output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    # output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    # output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    # output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    # output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    images_with_latent = T.concatenate([images, output], axis=1)

    X_v, X_h = next_stacks(images_with_latent, images_with_latent, N_CHANNELS + DIM_1, "Dec.PixInput", filter_size = 7, hstack = "hstack_a", residual = False)

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h,  DIM_PIX, "Dec.Pix"+str(i+1), filter_size = 3)


    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_1, filter_size=1, inputs=X_h))
    # output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=X_h))
    # skip_outputs.append(output)

    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    # output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_1, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_no_blind_z_everywhere(latents, images):
    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    X_v, X_h = next_stacks(images, images, N_CHANNELS, "Dec.PixInput", global_conditioning = output, filter_size = 7, hstack = "hstack_a")

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h, DIM_PIX, "Dec.Pix"+str(i+1), global_conditioning = output, filter_size = 3)
        if i > 0:
            X_h = X_h + prev_X_h
        prev_X_h = X_h

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=X_h))
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

# def auto_regress_generation(fn, latents, images):
#     samples = T.zeros_like(images)

#     for j in xrange(HEIGHT):
#         for k in xrange(WIDTH):
#             for i in xrange(N_CHANNELS):
#                 T.inc_subtensor()
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

def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (
        np.random.uniform(size=images.shape) < images
    ).astype(theano.config.floatX)

# def generate_with_only_receptive_field(fn, samples, latents):
#     h, w =  HEIGHT, WIDTH
#     receptive_field = 3 + ((PIXEL_CNN_FILTER_SIZE/2)*PIXEL_CNN_LAYERS)

#     next_samples = samples

#     t0 = time.time()
#     for j in xrange(HEIGHT):
#         for k in xrange(WIDTH):
#             for i in xrange(N_CHANNELS):
#                 j_min, j_end, j_res, k_min, k_end, k_res = get_receptive_area(h, w, receptive_field, j,k)
#                 res = fn(latents, next_samples[:,:,j_min:j_end, k_min:k_end])
#                 next_samples[:, i, j, k] = binarize(res[:, i, j_res, k_res])
#                 samples[:, i, j, k] = res[:, i, j_res, k_res]

#     t1 = time.time()
#     print("With only receptive field time taken is {:.4f}s".format(t1 - t0))

#     return samples




def Decoder_no_blind_conditioned_on_z(latents, images):
    output = latents

    X_v, X_h = next_stacks_gated(
                images, images, N_CHANNELS, "Dec.PixInput",
                global_conditioning = latents, filter_size = 7,
                hstack = "hstack_a", residual = False
                )

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks_gated(X_v, X_h, DIM_PIX, "Dec.Pix"+str(i+1), global_conditioning = latents, filter_size = PIXEL_CNN_FILTER_SIZE)


    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=X_h)
    output = PixCNN_condGate(output, latents, DIM_PIX, name='Dec.PixOut1.cond' )
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = PixCNN_condGate(output, latents, DIM_PIX, name='Dec.PixOut2.cond' )
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_condRMB(latents, images):
    output = latents

    _, X_next = next_stacks_gated(
                images, images, N_CHANNELS, "Dec.PixInput",
                global_conditioning = latents, filter_size = 7,
                hstack = "hstack_a", residual = False
                )

    for i in xrange(PIXEL_CNN_LAYERS):
        X_next = condRMB( "Dec.pixCNN_RMB_{}".format(i+1), X_next, DIM_PIX, cond = latents, filter_size = PIXEL_CNN_FILTER_SIZE)


    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=X_next)
    output = PixCNN_condGate(output, latents, DIM_PIX, name='Dec.PixOut1.cond' )
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = PixCNN_condGate(output, latents, DIM_PIX, name='Dec.PixOut2.cond' )
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output

def Decoder_no_blind_conditioned_on_z_skip(latents, images):
    output = latents
    skip_sum = floatX(0.)

    X_v, X_h, to_skip = next_stacks_gated_skip(
                images, images, N_CHANNELS, "Dec.PixInput",
                global_conditioning = latents, filter_size = 7,
                hstack = "hstack_a", residual = False
                )

    skip_sum += to_skip

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h, to_skip = next_stacks_gated_skip(X_v, X_h, DIM_PIX, "Dec.Pix"+str(i+1), global_conditioning = latents, filter_size = PIXEL_CNN_FILTER_SIZE)
        skip_sum += to_skip

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=X_h)
    output = T.nnet.relu(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=2*DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=T.concatenate([output, skip_sum/floatX(PIXEL_CNN_LAYERS + 1)], axis = 1))
    output = T.nnet.relu(output)


    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=2*DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = T.nnet.relu(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut4', input_dim=2*DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)

    return output



def Decoder_no_blind_z_bias(latents, images):

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=DIM_1, inputs=latents)
    output = output[:, :, None, None]

    X_v, X_h = next_stacks(images, images, N_CHANNELS, "Dec.PixInput", filter_size = 7, hstack = "hstack_a")

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h, DIM_PIX, "Dec.Pix"+str(i+1), filter_size = 3)
        if i > 0:
            X_h = X_h + 0.5*prev_X_h + 0.5*output
        prev_X_h = X_h

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=X_h))
    # skip_outputs.append(output)

    # output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=1, inputs=output))
    # skip_outputs.append(output)

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)
    # output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=T.concatenate(skip_outputs, axis=1), he_init=False)

    return output


if args.decoder_algorithm == 'cond_z_bias':
    decode_algo = Decoder_no_blind_conditioned_on_z
elif args.decoder_algorithm == 'upsample_z_conv':
    decode_algo = Decoder_no_blind
elif args.decoder_algorithm == 'upsample_z_no_conv':
    decode_algo = Decoder_no_blind_vary_up_sampling
elif args.decoder_algorithm == 'vae_only':
    decode_algo = Decoder_only_vae
elif args.decoder_algorithm == 'cond_z_bias_skip':
    decode_algo = Decoder_no_blind_conditioned_on_z_skip
elif args.decoder_algorithm == 'traditional':
    decode_algo = Decoder_traditional
elif args.decoder_algorithm == 'traditional_exact':
    decode_algo = Decoder_traditional_exact
elif args.decoder_algorithm == 'condRMB':
    decode_algo = Decoder_condRMB
elif args.decoder_algorithm == 'upsample_z_conv_tied':
    decode_algo = Decoder_no_blind_tied_pixelCNN_weights
else:
    assert False, "you should never be here!!"


if args.encoder == 'simple':
    encoder = Encoder
elif args.encoder == 'with_elu':
    encoder = Encoder_with_elu
else:
    encoder = Encoder_complecated
"""
total_iters = T.iscalar('total_iters')
images = T.tensor4('images') # shape: (batch size, n channels, height, width)

mu, log_sigma = encoder(images)

if VANILLA:
    latents = mu
else:
    eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
    latents = mu + (eps * T.exp(log_sigma))

# Theano bug: NaNs unless I pass 2D tensors to binary_crossentropy
reconst_cost = T.nnet.binary_crossentropy(
    T.nnet.sigmoid(
        decode_algo(latents, images).reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
    ),
    images.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
).sum(axis=1)

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(
    mu,
    log_sigma
).sum(axis=1)

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + BETA*(alpha * reg_cost)

sample_fn_latents = T.matrix('sample_fn_latents')
sample_fn = theano.function(
    [sample_fn_latents, images],
    T.nnet.sigmoid(decode_algo(sample_fn_latents, images)),
    on_unused_input='warn'
)

get_reg_cost = theano.function([images], reg_cost.mean())

encode_fn = theano.function([images], mu)
"""
train_data, dev_data, test_data = lib.mnist_stochastic_binarized.load(
    BATCH_SIZE,
    TEST_BATCH_SIZE
)

#lib.load_params(args.file_to_load)
random_project_mat = np.random.random((784, LATENT_DIM))

def encode_fn(X):
    X = X.reshape((X.shape[0], -1))
    return np.dot(X, random_project_mat)


val_accuracy  = lib.latent_train_utils.train_svm(encode_fn, train_data, dev_data, dimension = LATENT_DIM)
"""
reg_costs = []
for (images, targets) in dev_data():
    reg_costs.append(get_reg_cost(images))

print "KL cost is {}".format(np.mean(reg_costs))
"""
######################## Debugging SVM ################


