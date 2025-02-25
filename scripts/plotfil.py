import sys

import numpy as np
import matplotlib.pylab as plt
from blimpy import Waterfall

fil_file = sys.argv[1]

wf = Waterfall(fil_file, t_start=0, t_stop=int(sys.argv[2]))# Access header information
header = wf.header
print(header)

# Access the actual data
data = wf.data
np.save('here.npy', data[:,0,:])
print(np.sum(data))
fig = plt.figure()
plt.imshow(data[:,0,:])
plt.show()

wf.plot_waterfall()
