import os

import argparse

parser = argparse.ArgumentParser(description='Generating images pixel by pixel')

add_arg = parser.add_argument

add_arg('-v', '--vary', required = True)

args = parser.parse_args()

assert(args.vary in ['beta1', 'beta2', 'eps', 'latent_dim', 'dim_pix', 'nothing' ])

cluster_command = 'jobdispatch --gpu --queue=gpu_8 --duree=20:00:00 --repeat_jobs=4 --raw="#PBS -l nodes=1:gpus=4" --project=jvb-000-ag'
cmd = "python -u experiments/tf_vae_pixel/resnet.py"

eps = [ '0.00000001', '0.000001', '0.001', '0.1']
beta1 = ['0.9', '0.75', '0.99']

beta2 = ['0.99', '0.75', '0.9']

vary_arr_dict = {'eps': eps, 'beta1':beta1, 'beta2':beta2}

params = {'eps': '0.000001', 'beta1':'0.9', 'beta2':'0.99'}

vary_arr = vary_arr_dict[args.vary]


for i,val in enumerate(vary_arr):
    params[args.vary] = val

    FLAGS = " --eps {eps} --beta1 {beta1} --beta2 {beta2} --num_gpu 4".format(**params)
    final_command = "{} {} {}".format(cluster_command, cmd, FLAGS)
    print final_command
    # os.system(final_command)