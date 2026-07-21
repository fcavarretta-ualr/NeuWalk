#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from morphgenpy.io import read_swc
from morphgenpy.profiles import NeuriteProfile
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
    args = parser.parse_args()
  
    basal_dendrites = [extract_neurites(m['morphology'], section_type="basal_dendrite") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "apical_dendrite"]) if len(m['morphology'])]
    apical_dendrites = [extract_neurites(m['morphology'], section_type="apical_dendrite") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "basal_dendrite"]) if len(m['morphology'])]
    apical_obliques = [extract_neurites(m['morphology'], section_type="apical_oblique") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "basal_dendrite"]) if len(m['morphology'])]
    apical_dendrites_with_oblique = [extract_neurites(m['morphology'], section_type="apical_dendrite") for m in load_morphologies(args.directory, delete_section_types=["unknown", "apical_secondary_oblique", "apical_secondary_dendrite"]) if len(m['morphology'])]
      
    stats = {}

    stats['basal_dendrite'] = extract_statistics(basal_dendrites, bin_size=args.bin_size)
    stats['apical_dendrite'] = extract_statistics(apical_dendrites, bin_size=args.bin_size)
    stats['apical_dendrite_with_oblique'] = extract_statistics(apical_dendrites_with_oblique, bin_size=args.bin_size)
    stats['apical_oblique'] = extract_statistics(apical_obliques, bin_size=args.bin_size)

    bifurcation_internal_density = stats['apical_dendrite_with_oblique']['bifurcation_internal_density']
    print("bifurcation_internal_density=", bifurcation_internal_density)
    
    profiles = {}
    for section_type, params in stats.items():
        if section_type in ['apical_dendrite_with_oblique']:
          continue
        
        del params['total_length']
        del params['bifurcation_internal_density']
          
        rng = Random(args.seed)
        topol_synthesizer = TopologySynthesizer(rng=rng, step_size=args.step_size, bin_size=args.bin_size, section_type=section_type, **params)
        topol_synthesizer.synthesize_progressive(n_std=2)

        if section_type == "apical_oblique":
          profiles[section_type] = topol_synthesizer.roots
          continue
        
        # for non oblique, roots are attached to the soma          
        soma = NeuriteProfile(step_size=args.step_size, section_type="soma")
        for root in topol_synthesizer.roots:
            root.connect(soma, relation="parent")

        # store profile
        profiles[section_type] = soma

        
####        
######    import assign_neurites as an
######    bin_size = 10
######    rng = misc.Random(args.seed)
######    ret = an.assign_neurites_to_bins(rng, int(profiles['apical_oblique'].sholl_plot(bin_size)[0]), bifurcation_internal_density)
######    
######
######
######    to_connect = []
######    iroot = 0
######    for n_internal, bin_index in ret:
######
######      for i in range(n_internal):
######        assign = an.neurites_by_distance_bin(profiles['apical_dendrite'].children, bin_index * bin_size, (bin_index + 1) * bin_size)
######
######        index = int(rng.random() * len(assign))
######
######        neurite = assign[index]['neurite']
######        interval = assign[index]['interval']
######        an.cut_neurite(rng, neurite, interval)
######
######        oblique_dendrite = profiles['apical_oblique'].children[iroot]
######        oblique_dendrite.internal_bifurcation = True
######        for _oblique_dendrite in oblique_dendrite._iter_sections():
######          _oblique_dendrite.order = neurite.order + 1
######        neurite.connect(oblique_dendrite, relation="child")
######        iroot += 1
####        
##        
    b1 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (170., 0., 0.), (2.5, 2.5), (2.5, 2.5), 1, 2, strict=True)
    b2 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 170.]), (310., 0., 0.), (2.5, 2.5), (25, 50.0), 1, 2, strict=True)
    b3 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 480.]), (260., 0., 0.), (25, 50.0), (25.0, 100.0), 1, 2, strict=True)
    b4 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 740.]), (260., 0., 0.), (25.0, 100.0), (50.0, 200.0), 1, 2, strict=True)
    b = b1 + b2 + b3 + b4

    elongation_bias = []
    elongation_bias.append((0.003, biases.get_elongation("sibling_repulsion", 10.0, -2)))
    elongation_bias.append((0.006, biases.get_elongation("nonrelated_repulsion", 10.0, -2)))
    elongation_bias.append((0.2, b))
    elongation_bias.append((0.175, biases.get_elongation("plane_boundary", np.array([0., 0., 900.]), (np.pi, 0.), 20, -2)))
    elongation_random_weight = 0.75
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 6)

    morph_synthesizer = MorphologySynthesizer(
        root=profiles['apical_dendrite'],
        rng=rng,
        theta=0,
        phi=0,
        axis_direction=np.array([0.0, 0.0, 1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
    )
    
    soma_apical = morph_synthesizer.synthesize()
    
    
##    elongation_bias = list()
##    elongation_bias.append((0.003, biases.get_elongation("sibling_repulsion", 10.0, -2)))
##    elongation_bias.append((0.006, biases.get_elongation("nonrelated_repulsion", 10.0, -2)))
##    elongation_bias.append((0.001, biases.get_elongation("root_repulsion", None, None)))
##
##    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 6)
##    
##    morph_synthesizer = MorphologySynthesizer(
##        root=profiles['apical_oblique'],
##        rng=rng,
##        theta=np.pi / 2,
##        phi=(0, 2 * np.pi),
##        axis_direction=np.array([0.0, 0.0, -1.0]),
##        bifurcation_bias=bifurcation_bias,
##        elongation_bias=elongation_bias,
##        elongation_random_weight=elongation_random_weight
##    )
##    soma_basal = morph_synthesizer.synthesize()
    
    
    elongation_bias = list()
    elongation_bias.append((0.003, biases.get_elongation("sibling_repulsion", 10.0, -2)))
    elongation_bias.append((0.006, biases.get_elongation("nonrelated_repulsion", 10.0, -2)))
    elongation_bias.append((0.001, biases.get_elongation("root_repulsion", None, None)))

    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 6)
    
    morph_synthesizer = MorphologySynthesizer(
        root=profiles['basal_dendrite'],
        rng=rng,
        theta=(np.pi / 6, np.pi * 5 / 6),
        phi=(0, 2 * np.pi),
        axis_direction=np.array([0.0, 0.0, -1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight
    )
    soma_basal = morph_synthesizer.synthesize()

    for r in soma_basal.children:
        r.disconnect(soma_basal)
        r.connect(soma_apical, relation="parent")
        
    plot_morphology(soma_apical)

if __name__ == "__main__":
    main()
