#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('recons_and_kl.pdf')

recons_costs = [
                61.35,
                69.95,
                71.82,
                71.79,
                72.16,
                72.71,
                73.24,
                73.52,
                73.47,
                73.47,
                73.63,
                73.86,
                74.09,
                74.24,
                75.01,
               ]

kl_costs = [
            26.33,
            11.59,
            9.4,
            9.32,
            8.65,
            8.12,
            7.55,
            7.19,
            7.29,
            7.2,
            7.06,
            6.86,
            6.56,
            6.38,
            5.3,         
           ]

for i in range(len(kl_costs)):
    print kl_costs[i] + recons_costs[i]

N = len(kl_costs)
ind = np.arange(N)    # the x locations for the groups
width = 0.7
spacing = 0.3

p1 = plt.bar(ind + spacing, kl_costs, width, color='b')
p2 = plt.bar(ind + spacing, recons_costs, width, color='y',
             bottom=kl_costs)

plt.ylabel('Cost', fontsize = 20)
plt.xlabel('#PixelCNN layers', fontsize = 20)

plt.xticks(ind + width/2 + spacing, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '21'), fontsize = 15)
plt.yticks(fontsize = 15)
plt.tick_params(axis=u'x', which=u'x',length=0)

plt.legend((p1[0], p2[0]), ('KL-divergence', 'Reconstruction'),
            bbox_to_anchor = (0., 1.02, 1., .102), loc = 1,
            ncol = 2, mode = 'expand', borderaxespad = 0.,
            fontsize = 20)

#plt.show()
#plt.savefig('recons_and_kl.png', transparent = True)
#plt.close()

plt.savefig(pp, format = 'pdf')
pp.close()
