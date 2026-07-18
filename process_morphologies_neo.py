from morphgenpy.analysis import morphologies
from morphgenpy.io import swc
import morphgenpy.visualization as vis

import sys

delete_section_types = ["unknown", "axon", "apical_secondary_dendrite", "apical_secondary_oblique"]

m = morphologies.load_morphologies(sys.argv[-1], delete_section_types=delete_section_types)


for x in m:
  print(x)
  vis.plot_morphology(x['morphology'])

  print(x['filename'])


  swc.write_swc(str(x['filename']) + '.2', x['morphology'])
