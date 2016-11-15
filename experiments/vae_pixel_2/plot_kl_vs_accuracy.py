import os
import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
results = {}
results["VAE"] = {"dim_pix_16": {"KL":[], "val":[]}, "dim_pix_32": {"KL":[], "val":[]}}
results["pixelVAE"] = {"dim_pix_16": {"KL":[], "val":[]}, "dim_pix_32": {"KL":[], "val":[]}}

for d in os.listdir("LOGS"):
	for f in os.path.join("LOGS", d):
		if "out" in f:
			current_log = os.path.joing("LOGS", d, f)
			dim_type = None
			KL = None
			valid_accuracy = None
			model_type = None
			with open(current_log, 'rb') as lg:
				lines = lg.readlines()
				for ln in lines:
					if "vae_only" in ln:
						model_type = "VAE"

					if "cond_z_bias" in ln:
						model_type = "pixelVAE"

					if "KL cost" in ln:
						KL = float(ln.split()[-1])

					if "Evaluation" in ln:
						valid_accuracy = float(ln.split()[-1])

					if "dim_pix_16" in ln:
						dim_type = "dim_pix_16"

					if "dim_pix_32" in ln:
						dim_type = "dim_pix_32"

					results[model_type][dim_type]["KL"].append(KL)
					results[model_type][dim_type]["val"].append(valid_accuracy)


plt.plot(results["VAE"]["dim_pix_16"]["KL"], results["VAE"]["dim_pix_16"]["val"], "r.-", label="VAE-16 dim")
plt.plot(results["VAE"]["dim_pix_32"]["KL"], results["VAE"]["dim_pix_32"]["val"], "g.-", label="VAE-64 dim")
plt.plot(results["pixelVAE"]["dim_pix_16"]["KL"], results["pixelVAE"]["dim_pix_16"]["val"], "r*-", label="pixelVAE-16 dim")
plt.plot(results["pixelVAE"]["dim_pix_32"]["KL"], results["pixelVAE"]["dim_pix_32"]["val"], "g*-", label="pixelVAE-64 dim")

plt.xlabel("KL cost")
plt.ylabel("Validation accuracy")

plt.legend()

plt.savefig("plot_kl_vs_accuracy.jpg")
plt.close()

