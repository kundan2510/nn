import os
import argparse

parser = argparse.ArgumentParser(description='Generating images pixel by pixel')

add_arg = parser.add_argument

add_arg('-v', '--vary', required = True)
add_arg('-lb',"--log_base_name", required = True)
add_arg('-gpu',"--gpu_num", default=0, type=int)

args = parser.parse_args()

theano_flag = "THEANO_FLAGS=mode=FAST_RUN,device=gpu{},floatX=float32,lib.cnmem=0.95".format(args.gpu_num)
cmd = "python -u experiments/vae_pixel_2/mnist.py"

assert(args.vary in ['alpha_iters', 'num_layers', 'filter_sizes', 'latent_dim', 'dim_pix' ])

alpha_iters = [6000, 8000, 10000, 12000, 14000]
num_layers = [6, 8, 10, 12, 14]
filter_sizes = [3,5,7]
latent_dim = [32, 48, 64, 80, 96]
dim_pix = [16, 32, 48, 64]

vary_arr_dict = {'alpha_iters': alpha_iters, 'num_layers':num_layers, 'filter_sizes':filter_sizes, 'latent_dim':latent_dim, 'dim_pix':dim_pix}

vary_arr = vary_arr_dict[args.vary]

params = {'alpha_iters': 10000, 'num_layers':10, 'filter_sizes':5, 'latent_dim':64, 'dim_pix':32, 'algo': 'cond_z_bias'}

for i,val in enumerate(vary_arr):
    params[args.vary] = val

    FLAGS = "-L {num_layers} -ait {alpha_iters} -fs {filter_sizes} -ldim {latent_dim} -dpx {dim_pix} -algo {algo}".format(**params)
    final_command = "{} {} {} > {}_{}.log".format(theano_flag, cmd, FLAGS, args.log_base_name, i)
    print final_command
    os.system(final_command)

