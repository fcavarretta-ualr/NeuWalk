from morphgenpy.analysis import morphologies
from morphgenpy.io import swc
import morphgenpy.visualization as vis

import numpy as np

import sys

delete_section_types = ["unknown", "axon", "soma"]

directory = sys.argv[-2] 
section_type = sys.argv[-1]

if section_type == "basal_dendrite":
  delete_section_types.append("apical_dendrite")
elif section_type == "apical_dendrite":
  delete_section_types.append("basal_dendrite")
else:
  print(f"Unknown section type {section_type}")

total = []
for m in morphologies.load_morphologies(directory, delete_section_types=delete_section_types):
  s = sum(r.total_length  for r in m if r.section_type == section_type)
  if np.isnan(s):
    continue
  total.append(s)
  
print(len(total), np.mean(total), np.std(total) / np.sqrt(len(total)))
