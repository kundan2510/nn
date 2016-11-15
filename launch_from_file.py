import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Generating images pixel by pixel')

add_arg = parser.add_argument

add_arg('--use_k80',default=None, required=False)
add_arg('--file', required=True)

args = parser.parse_args()

theano_flag = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.95"
if args.use_k80 is not None:
	k80_command = '--raw="#PBS -l feature=k80"'
else:
	k80_command = ""

cluster_command = "jobdispatch --gpu --duree=23:59:00 {} --env={} --project=jvb-000-ag".format(k80_command, theano_flag)

with open(args.file, 'rb') as f:
	cmds = f.read().split('\n')
	for cmd in cmds:
		if len(cmd) > 5:
			full_cmd = '{} {}'.format(cluster_command, cmd)
			print full_cmd
			os.system(full_cmd)
