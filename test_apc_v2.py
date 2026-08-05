#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from neuwalk.io import read_swc
from neuwalk.core.topology import SectionSynthesizer, connect_internal_branches
from neuwalk.synthesis.topology.sampling import EventSampler
from neuwalk.synthesis import TopologySynthesizer, MorphologySynthesizer
from neuwalk.visualization import plot_morphology
import neuwalk.synthesis.morphology.biases as biases
from neuwalk.analysis.morphologies import load_morphologies
from neuwalk.analysis.extraction import extract_statistics
from neuwalk.random import Random


import json

def merge_profiles(profile_roots):
    # for non oblique, roots are attached to the soma          
    profile_soma = SectionSynthesizer(1, label="soma")
    for root in profile_roots:
        root.connect(profile_soma, relation="parent")
    return profile_soma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--n-std", type=float, default=1.0)
    parser.add_argument("--max-attempts-per-window", type=int, default=10)
    parser.add_argument("--max-total-attempts", type=int, default=1000)
    args = parser.parse_args()


    # extract statistics for basal, apical, and oblique sections
    # param contains the sections to be discarded
    discarded_sections = {
      'basal_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite"],
      'apical_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite"]
      }

    # stat contains the statistics
    all_params = {}
    profiles = {}
    for label, delete_labels in discarded_sections.items():
      print(f"Elaboration of {label}")
      
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
      
      print(f"\tGenerating Branching-and-annihilating profile...", end="")
      topol_synthesizer = TopologySynthesizer(
            Random(args.seed),
            step_size=args.step_size,
            bin_size=args.bin_size,
            label=label,
            **params
        )

      topol_synthesizer.synthesize_progressive(
          n_std=args.n_std,
          max_attempts_per_window=args.max_attempts_per_window,
          max_total_attempts=args.max_total_attempts,
          verbose=args.verbose
      )

      profiles[label] = topol_synthesizer.roots
      print("done\n")
      print("Summary")
      print("---------------------------------------------------------")
      topol_synthesizer.describe()
      print("---------------------------------------------------------\n\n")
      
    with open("parameters.json", "w") as file:
      json.dump(all_params, file, indent=4, default=lambda value: value.tolist())
    quit()
    
    # synthesize apical dendritic tree
    elongation_random_weight = 1
    max_angle = np.pi / 2
   

    # create self-avoidance bias
    r1 = biases.get_elongation("sibling_repulsion", 25.0, -2)
    r2 = biases.get_elongation("nonrelated_repulsion", 25.0, -2)
    r = r1 + r2
    
    elongation_bias = [ (0.25, biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)),
                        (0.005, r), (0.2, biases.get_elongation("root_repulsion", 750.0, -2)) ]
    
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)

    # initializa the synthesizer for apical sections
    apic_synthesizer = MorphologySynthesizer(
        root=merge_profiles(profiles['apical_dendrite']),
        rng=Random(seed=args.seed),
        theta=0,
        phi=0,
        axis_direction=np.array([0.0, 0.0, 1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )

    # synthesize apical sections
    apic_synthesizer.synthesize()

    elongation_bias = [ (0.25, biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (-1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)),
                        (0.005, r), (0.2, biases.get_elongation("root_repulsion", 750.0, -2)) ]     

    # synthesize apical sections        
    basal_synthesizer = MorphologySynthesizer(
        root=merge_profiles(profiles['basal_dendrite']),
        rng=Random(seed=args.seed),
        theta=(0, np.pi / 2),
        phi=(0, 2 * np.pi),
        axis_direction=np.array([0.0, 0.0, -1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )
    basal_synthesizer.synthesize(soma=apic_synthesizer.soma)


    # plot the morphology
    plot_morphology(basal_synthesizer.soma)

if __name__ == "__main__":
    main()
