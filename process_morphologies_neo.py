from neuwalk.analysis import morphologies
from neuwalk.io import swc
import neuwalk.visualization as vis

import sys

delete_categories = ["unknown", "axon", "apical_secondary_dendrite", "apical_secondary_oblique"]

m = morphologies.load_morphologies(sys.argv[-1], delete_categories=delete_categories)


for x in m:
  if x == "morphologies/Neocortex/PYR/C030397A-P2.CNG.swc":
    print(x)
    vis.plot_morphology(x['morphology'])

    print(x['filename'])


  swc.write_swc(str(x['filename']), x['morphology'])
