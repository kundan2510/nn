THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python experiments/vae_pixel_2/mnist.py -L 9 -algo cond_z_bias  -ot running_cond_z_bias 
