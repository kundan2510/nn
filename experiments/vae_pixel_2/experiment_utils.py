import re

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
					beta = float(re.split("=|,", info[1])[1])
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



