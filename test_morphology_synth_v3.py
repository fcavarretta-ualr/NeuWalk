#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from morphgenpy.io import read_swc
from morphgenpy.profiles import NeuriteProfile, connect_internal_branches
from morphgenpy.sampling import EventSampler
from morphgenpy.synthesis import TopologySynthesizer, MorphologySynthesizer
from morphgenpy.visualization import plot_morphology
import morphgenpy.biases as biases
from morphgenpy.analysis.morphologies import load_morphologies
from morphgenpy.analysis.extraction import extract_statistics
from morphgenpy.misc import Random

import morphgenpy.misc as misc

def translate_subtree(section, target=None):
  """ translate the points with the first one of the root coinciding with source """
  # translate the section
  section.points = misc.translate_points(section.points, section.points[0], target=target)

  # translate children and their subtrees
  for ch in section.children:
    translate_subtree(ch, target=section.points[-1])
    

def extract_neurites(morphology, section_type):
  """
  Clone a morphology and extract top-level trees of selected section types.

  A selected section becomes a retained root when it has no parent or when
  its parent's section type is different.
  """
  
  root_sections = []

  # find the roots
  for root in morphology:

    if root.section_type == "soma":
      for ch in root.children:
        if ch.section_type == section_type:
          ch.disconnect_from_parent()
          root_sections.append(ch)
    else:
      root_sections.append(root)
      
  return root_sections

    

def main():
    parser = argparse.ArgumentParser(
        description="Estimate event statistics and test EventSampler."
    )
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
  
    basal_dendrites = [extract_neurites(m['morphology'], section_type="basal_dendrite") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "apical_dendrite"]) if len(m['morphology'])]
    apical_dendrites = [extract_neurites(m['morphology'], section_type="apical_dendrite") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "basal_dendrite"]) if len(m['morphology'])]
    apical_obliques = [extract_neurites(m['morphology'], section_type="apical_oblique") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "basal_dendrite", "soma"]) if len(m['morphology'])]
    apical_dendrites_with_oblique = [extract_neurites(m['morphology'], section_type="apical_dendrite") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_secondary_oblique", "apical_secondary_dendrite"]) if len(m['morphology'])]
      
    stats = {}

    stats['basal_dendrite'] = extract_statistics(basal_dendrites, bin_size=args.bin_size)
    stats['apical_dendrite'] = extract_statistics(apical_dendrites, bin_size=args.bin_size)
    stats['apical_oblique'] = extract_statistics(apical_obliques, bin_size=args.bin_size)

    bifurcation_internal_density = extract_statistics(apical_dendrites_with_oblique, bin_size=args.bin_size)['bifurcation_internal_density']
    print("bifurcation_internal_density=", bifurcation_internal_density)
    
    profiles = {}
    for section_type, params in stats.items():
        print(f"Generating {section_type}...")
        topol_synthesizer = TopologySynthesizer(
            Random(args.seed),
            step_size=args.step_size,
            bin_size=args.bin_size,
            section_type=section_type,
            sholl_plot=params['sholl_plot'],
            bifurcation_count=params['bifurcation_count'],
            primary_count_range=params['primary_count_range'],
            no_bifurcation_bins=params['no_bifurcation_bins'],
            no_annihilation_bins=params['no_annihilation_bins']
        )

        topol_synthesizer.synthesize_progressive(
            n_std=args.n_std,
            max_attempts_per_window=args.max_attempts_per_window,
            max_total_attempts=args.max_total_attempts,
            verbose=args.verbose
        )


        profiles[section_type] = topol_synthesizer.roots

    # connect obliques
    connect_internal_branches(profiles['apical_oblique'], profiles['apical_dendrite'], Random(args.seed), bifurcation_internal_density, args.bin_size)
    
    b1 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (220., 0., 0.), (2.5, 2.5), (2.5, 2.5), 1, 1, strict=True)
    b2 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 220.]), (50., 0., 0.), (2.5, 2.5), (25.0, 75.0), 1, 1, strict=True)
    b3 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 270.]), (470., 0., 0.), (25.0, 75.0), (25.0, 75.0), 1, 1, strict=True)
    b4 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 740.]), (160., 0., 0.), (25.0, 75.0), (150.0, 150.0), 1, 1, strict=True)
    b5 = biases.get_elongation("plane_boundary", np.array([0., 0., 900.]), (np.pi, 0.), 10, -2)
    b = b1 + b2 + b3 + b4 + 2 * b5

    c1 = biases.get_elongation("sibling_repulsion", 25.0, -2)
    c2 = biases.get_elongation("nonrelated_repulsion", 25.0, -2)
    c = c1 + c2
    elongation_bias = []
    elongation_bias.append((0.5, b))
    elongation_bias.append((0.002, c))
    elongation_random_weight = 5
    max_angle = np.pi / 4
    
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)
    bifurcation_internal_bias = biases.get_bifurcation("internal_branch", np.pi / 2)  

    

    # for non oblique, roots are attached to the soma          
    soma_profile = NeuriteProfile(step_size=args.step_size, section_type="soma")
    for root in profiles['apical_dendrite']:
        root.connect(soma_profile, relation="parent")
        
    morph_synthesizer = MorphologySynthesizer(
        root=soma_profile,
        rng=misc.Random(seed=args.seed),
        theta=0,
        phi=0,
        axis_direction=np.array([0.0, 0.0, 1.0]),
        bifurcation_bias=bifurcation_bias,
        bifurcation_internal_bias=bifurcation_internal_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )
    
    soma_apical = morph_synthesizer.synthesize()

    elongation_bias = list()
    r = biases.get_elongation("root_repulsion", 100, -1)
    elongation_bias.append((0.5, r))
    elongation_bias.append((0.002, c))

    # generate the second order
    morph_synthesizer.synthesize(
      is_root_like=True,
      elongation_bias=elongation_bias,
      )
    
  


    soma_profile = NeuriteProfile(step_size=args.step_size, section_type="soma")
    for root in profiles['basal_dendrite']:
        root.connect(soma_profile, relation="parent")
        
    morph_synthesizer = MorphologySynthesizer(
        root=soma_profile,
        rng=misc.Random(seed=args.seed),
        theta=(np.pi / 6, np.pi * 5 / 6),
        phi=(0, 2 * np.pi),
        axis_direction=np.array([0.0, 0.0, -1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )
    soma_basal = morph_synthesizer.synthesize()

    for r in soma_basal.children:
        r.disconnect(soma_basal)
        r.connect(soma_apical, relation="parent")
        
    plot_morphology(soma_apical)

if __name__ == "__main__":
    main()
