from neuwalk.analysis import morphologies
from neuwalk.io import swc
import neuwalk.visualization as vis

import numpy as np

import sys

delete_labels = ["unknown", "axon", "soma"]

directory = sys.argv[-2] 
label = sys.argv[-1]

if label == "basal_dendrite":
  delete_labels.append("apical_dendrite")
elif label == "apical_dendrite":
  delete_labels.append("basal_dendrite")
else:
  print(f"Unknown label {label}")

total = []
for m in morphologies.load_morphologies(directory, delete_labels=delete_labels):
  s = sum(r.total_length  for r in m if r.label == label)
  if np.isnan(s):
    continue
  total.append(s)
  
print(len(total), np.mean(total), np.std(total) / np.sqrt(len(total)))
