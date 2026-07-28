#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from neuwalk.io import read_swc
from neuwalk.profiles import NeuriteProfile, connect_internal_branches
from neuwalk.sampling import EventSampler
from neuwalk.synthesis import TopologySynthesizer, MorphologySynthesizer
from neuwalk.visualization import plot_morphology
import neuwalk.biases as biases
from neuwalk.analysis.morphologies import load_morphologies
from neuwalk.analysis.extraction import extract_statistics
from neuwalk.misc import Random

import neuwalk.misc as misc


def merge_profiles(profile_roots):
    # for non oblique, roots are attached to the soma          
    profile_soma = NeuriteProfile(1, section_type="soma")
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


    # extract statistics for basal, apical, and oblique dendrites
    # param contains the sections to be discarded
    discarded_sections = {
      'basal_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite"],
      'apical_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite"]
      }

    # stat contains the statistics

    profiles = {}
    for section_type, delete_section_types in discarded_sections.items():
      print(f"Elaboration of {section_type}")
      
      print(f"\tExtracting statistics...", end="")
      params = extract_statistics(
        load_morphologies(args.directory, delete_section_types=delete_section_types),
        bin_size=args.bin_size)
      print("done")

      # these params are not used for generation
      params.pop("total_length", None)
      params.pop("bifurcation_internal_density", None)

      
      print(f"\tGenerating Branching-and-annihilating profile...", end="")
      topol_synthesizer = TopologySynthesizer(
            Random(args.seed),
            step_size=args.step_size,
            bin_size=args.bin_size,
            section_type=section_type,
            **params
        )

      topol_synthesizer.synthesize_progressive(
          n_std=args.n_std,
          max_attempts_per_window=args.max_attempts_per_window,
          max_total_attempts=args.max_total_attempts,
          verbose=args.verbose
      )

      profiles[section_type] = topol_synthesizer.roots
      print("done\n")
      print("Summary")
      print("---------------------------------------------------------")
      topol_synthesizer.describe()
      print("---------------------------------------------------------\n\n")

    # synthesize apical dendritic tree
    elongation_random_weight = 5
    max_angle = np.pi / 4

    
    
   

    # create self-avoidance bias
    r1 = biases.get_elongation("sibling_repulsion", 25.0, -2)
    r2 = biases.get_elongation("nonrelated_repulsion", 25.0, -2)
    r = r1 + r2
    
    elongation_bias = [ (0.5, biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)),
                        (0.002, r) ]
    
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)

    # initializa the synthesizer for apical dendrites
    apic_synthesizer = MorphologySynthesizer(
        root=merge_profiles(profiles['apical_dendrite']),
        rng=misc.Random(seed=args.seed),
        theta=0,
        phi=0,
        axis_direction=np.array([0.0, 0.0, 1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )

    # synthesize apical dendrites
    apic_synthesizer.synthesize()


    # generate the second order (i.e., oblique dendrites)

    elongation_bias = [ (0.5, biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (1100., np.pi, 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)),
                        (0.002, r) ]
     

    # synthesize apical dendrites        
    basal_synthesizer = MorphologySynthesizer(
        root=merge_profiles(profiles['basal_dendrite']),
        rng=misc.Random(seed=args.seed),
        theta=(0, np.pi / 2),
        phi=(0, 2 * np.pi),
        axis_direction=np.array([0.0, 0.0, -1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )
    basal_synthesizer.synthesize()

    soma = apic_synthesizer.soma.clone()
    
    # unified basal and apical dendrites
    for r in basal_synthesizer.soma.children + apic_synthesizer.soma.children:
        r.disconnect_from_parent()
        r.connect(soma, relation="parent")


    # plot the morphology
    plot_morphology(soma)

if __name__ == "__main__":
    main()
