import sys

from blimpy import Waterfall

fil_file = sys.argv[1]

wf = Waterfall(fil_file, t_start=0, t_stop=int(sys.argv[2]))# Access header information
header = wf.header
print(header)

# Access the actual data
data = wf.data
print(data.shape)

wf.plot_waterfall()
