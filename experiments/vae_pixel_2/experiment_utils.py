import re
import numpy

def find_best_valid_iter_from_log(log_file_path):
	with open(log_file_path, 'rb') as f:
		lines = f.readlines()
		i = -1
		take_seriously = False

		best_valid_cost = 50000.
		best_valid_iter = 0

		while(i < (len(lines) -1)):
			i += 1
			curr_line = lines[i].rstrip('\n')
			if take_seriously is False :
				if "Namespace" in curr_line:
					info = curr_line.split()
					beta = numpy.float32(re.split("=|,", info[1])[1])
					decoder_algorithm = re.split("=|,|'", info[2])[2]
					dim_pix = int(re.split("=|,", info[3])[1])

					filter_size = int(re.split("=|,", info[5])[1])
					latent_dim = int(re.split("=|,", info[6])[1])
					num_layers = int(re.split("=|\)", info[7])[1])

					params = {
								"beta" : beta,
								"decoder_algorithm" : decoder_algorithm,
								"dim_pix" : dim_pix,
								"filter_size" : filter_size,
								"latent_dim" : latent_dim,
								"num_layers" : num_layers
							}
				if "epoch" not in curr_line:
					continue
				else:
					take_seriously = True

			if take_seriously is True and "epoch" in curr_line:
				info = curr_line.split()
				# import ipdb; ipdb.set_trace()
				curr_iter = int(info[1].split(':')[1])
				valid_cost = float(info[11].split(':')[1])
				alpha = float(info[13].split(':')[1])
				if alpha < 1.:
					continue
				else:
					if valid_cost < best_valid_cost:
						best_valid_cost = valid_cost
						best_valid_iter = curr_iter


		if best_valid_cost != 50000.:
			print "Best valid cost {} occured at iters {}".format(best_valid_cost, best_valid_iter)

	return params, best_valid_iter


def get_checkpoint_path(log_file_path):
	params, iters = find_best_valid_iter_from_log(log_file_path)

	if os.path.isdir('/home/kundan'):
		"""
		It is a cluster
		"""
		out_dir_prefix = '/home/kundan/mnist_saves_beta_new'
	else:
		"""
		It is a lab machine.
		"""
		out_dir_prefix = '/Tmp/kumarkun/mnist_saves_beta_new'


	DIR = out_dir_prefix + "/num_layers_" + str(params['num_layers']) + \
			params['decoder_algorithm']+ "_simple/dim_pix_" + \
			str(params['dim_pix']) + "_latent_dim_" + str(params['latent_dim']) + "/beta_" + str(params[beta]) + \
			"_fs_" + str(params['filter_size']) + "_alpha_iters_6000"

	if not os.path.exists(DIR):
		raise ValueError("Why does path {} not exist?".format(DIR))
	else:
		potential_checkpoints = []
		for f in os.listdir(DIR):
			if "iters{}".format(iters) in f:
				potential_checkpoints.append(f)

		if len(potential_checkpoints) != 1:
			raise ValueError("More that one potential checkpoints!!")
		else:
			checkpoint = os.path.join(DIR, potential_checkpoints[0])
			params['checkpoint'] = checkpoint

		return "-L {num_layers} -fs {filter_size} -algo {algo} -dpx {dim_pix} -ldim {latent_dim} -beta {beta} -file_to_load {file_to_load}".format(**params)


def get_all_evaluate_commands(log_folder):
	commands = []

	for d in os.listdir(log_folder):
		if "python" in d:
			for f in os.listdir(os.path.join(log_folder, d)):
				if "out" in f:
					curr_log = os.path.join(log_folder, d, f)
					print curr_log
					curr_params_str = get_checkpoint_path(curr_log)
					curr_cmd = "python experiments/vae_pixel_2/mnist_with_beta_evaluate_latent.py {}".format(curr_params_str)
					commands.append(curr_cmd)

	with open("evaluation_command.cmd", 'rb') as f:
		f.write("\n".join(commands))

if __name__ == "__main__":
	get_all_evaluate_commands('LOGS')

