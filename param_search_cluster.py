import os
import argparse
import numpy

parser = argparse.ArgumentParser(description='Generating images pixel by pixel')

add_arg = parser.add_argument

add_arg('-v', '--vary', required = True)

args = parser.parse_args()

theano_flag = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.95"
cluster_command = "jobdispatch --gpu --duree=20:00:00 --env={} --project=jvb-000-ag".format(theano_flag)
cmd = "python -u experiments/vae_pixel_2/{}"

assert(args.vary in ['alpha_iters', 'beta', 'num_layers', 'filter_sizes', 'latent_dim', 'dim_pix', 'nothing' ])

if args.vary == "beta":
    cmd = cmd.format("mnist_with_beta.py")
else:
    cmd = cmd.format("mnist.py")


alpha_iters = [5000, 6000, 7000]
num_layers = [12, 14, 16, 18, 20, 24]
filter_sizes = [5,7]
latent_dim = [4, 8, 12, 16, 20, 24]
dim_pix = [4, 8, 12, 16, 20, 24]
beta = numpy.arange(1, 5, 0.5)

#beta = [b for b in beta] + [1./b for b in beta]

beta = [1./b for b in beta]

vary_arr_dict = {'alpha_iters': alpha_iters, 'num_layers':num_layers, 'filter_sizes':filter_sizes, 'latent_dim':latent_dim, 'dim_pix':dim_pix, 'beta' : beta}

params = {'alpha_iters': 6000, 'num_layers':12, 'filter_sizes':5, 'latent_dim':64, 'dim_pix':32, 'algo': 'cond_z_bias', 'beta' : 1}

if args.vary == "nothing":
    for i in range(5):
        FLAGS = "-L {num_layers} -ait {alpha_iters} -fs {filter_sizes} -ldim {latent_dim} -dpx {dim_pix} -algo {algo} -beta {beta}".format(**params)
        final_command = "{} {} {}".format(cluster_command, cmd, FLAGS)
        print final_command
        os.system(final_command)
    exit(0)

vary_arr = vary_arr_dict[args.vary]


for i,val in enumerate(vary_arr):
    params[args.vary] = val

    FLAGS = "-L {num_layers} -ait {alpha_iters} -fs {filter_sizes} -ldim {latent_dim} -dpx {dim_pix} -algo {algo} -beta {beta}".format(**params)
    final_command = "{} {} {}".format(cluster_command, cmd, FLAGS)
    print final_command
    os.system(final_command)

