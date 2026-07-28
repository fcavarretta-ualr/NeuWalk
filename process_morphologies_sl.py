from neuwalk.analysis import morphologies
from neuwalk.io import swc
import neuwalk.visualization as vis

import sys

delete_section_types = ["unknown", "axon", "basal_dendrite"]

m = morphologies.load_morphologies(sys.argv[-1], delete_section_types=delete_section_types)


for x in m:
  print(x)
  vis.plot_morphology(x['morphology'])

  print(x['filename'])


  swc.write_swc(x['filename'], x['morphology'])
