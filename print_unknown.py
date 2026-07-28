from neuwalk.analysis import morphologies
from neuwalk.io import swc
import neuwalk.visualization as vis

import numpy as np

import sys

from pathlib import Path

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

files = sorted(Path(directory).rglob("*.swc"))

for filename, m1 in zip(files, morphologies.load_morphologies(directory, delete_section_types="axon")):
  s1 = sum(rr.length  for r in m1 for rr in r.subtree if rr.section_type == "unknown")  
  s2 = sum(rr.length  for r in m1 for rr in r.subtree)
  print(filename, s1, s2, s1/s2)
  if np.isnan(s2):
    continue
  total.append(s1/s2)
  
print(len(total), np.mean(total), np.std(total) / np.sqrt(len(total)))
