from neuwalk.analysis import morphologies
from neuwalk.io import swc
import neuwalk.visualization as vis

import numpy as np

import sys

from pathlib import Path

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

files = sorted(Path(directory).rglob("*.swc"))

for filename, m1 in zip(files, morphologies.load_morphologies(directory, delete_labels="axon")):
  s1 = sum(rr.length  for r in m1 for rr in r.subtree if rr.label == "unknown")  
  s2 = sum(rr.length  for r in m1 for rr in r.subtree)
  print(filename, s1, s2, s1/s2)
  if np.isnan(s2):
    continue
  total.append(s1/s2)
  
print(len(total), np.mean(total), np.std(total) / np.sqrt(len(total)))
