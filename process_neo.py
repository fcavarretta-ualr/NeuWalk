import argparse
from pathlib import Path

import numpy as np


from neuwalk.analysis.morphologies import load_morphologies
from neuwalk.analysis.extraction import extract_statistics
from neuwalk.io.swc import write_swc

source = "morphologies/Neocortex/PYR"
dest = "morphologies/Neocortex/PYR_curated"

for filename, m in load_morphologies(source, delete_labels=["apical_oblique", "unknown",  "apical_secondary_oblique", "apical_secondary_dendrite"], return_file_names=True):
  filename = Path(filename).name

  write_swc(dest + "/" + filename, m)

  

  
