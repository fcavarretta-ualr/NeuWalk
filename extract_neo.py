#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from neuwalk.analysis.morphologies import load_morphologies
from neuwalk.analysis.extraction import extract_statistics

import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bin-size", type=float, default=10.0)
    args = parser.parse_args()


    # extract statistics for basal, apical, and oblique sections
    # param contains the sections to be discarded
    discarded_sections = {
      'basal_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite", "axon"],
      'apical_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
      'apical_oblique':["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"]
      }

    
    bifurcation_internal_density =  extract_statistics(
      load_morphologies(args.directory,
                        delete_labels=["unknown",  "apical_secondary_oblique", "apical_secondary_dendrite", "basal_dendrite", "soma", "axon"]),
      bin_size=args.bin_size)['bifurcation_internal_density']
    
    
    # stat contains the statistics
    all_params = {}
    profiles = {}
    for label, delete_labels in discarded_sections.items():
      print(f"Elaboration of {label}")

      try:
          print(f"\tExtracting statistics...", end="")
          params = extract_statistics(
            load_morphologies(args.directory, delete_labels=delete_labels),
            bin_size=args.bin_size)
          print("done")

          # these params are not used for generation
          params.pop("total_length", None)
          params.pop("bifurcation_internal_density", None)
            

          all_params[label] = params.copy()
          all_params[label]['bin_size'] = args.bin_size
      except ValueError:
        pass

    all_params['apical_dendrite']['bifurcation_internal_density'] = bifurcation_internal_density
    
    with open("neuwalk/presets/neocortex/pyramidal.parameters.json", "w") as file:
      json.dump(all_params, file, indent=4, default=lambda value: value.tolist())



if __name__ == "__main__":
    main()
