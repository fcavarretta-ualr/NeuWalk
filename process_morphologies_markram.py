from neuwalk.analysis import morphologies
from neuwalk.io import swc
import neuwalk.visualization as vis

import sys


path = '/home/francesco/morphologies-downloads/morphologies/swc'


s1 = set()

for fname, m in morphologies.load_morphologies(path, delete_labels=["unknown", "axon", "basal_dendrite", "apical_oblique"], return_file_names=True):
  assert len(m) == 1
  
  try:
    sp = m[0].sholl_plot(bin_size=10)
  except ValueError:
    print(fname)
    continue
  
  max_dist = (len(sp) - 1) * 10

  if max_dist >= 900:
    s1.add(fname.name)
##    new_path = path + '/selected/' + 
##    swc.write_swc(new_path, m)


s2 = set()

for fname, m in morphologies.load_morphologies(path, delete_labels=["unknown", "axon", "apical_dendrite", "apical_oblique"], return_file_names=True):
  assert len(m) == 1
  

  if m[0].total_length >= 900:
    s2.add(fname.name)
##    new_path = path + '/selected/' + 
##    swc.write_swc(new_path, m)

s = s1.intersection(s2)

for fname, m in morphologies.load_morphologies(path, delete_labels=["unknown", "axon"], return_file_names=True):
  if fname.name in s:
    new_path = path + '/selected/' + fname.name
    swc.write_swc(new_path, m)
