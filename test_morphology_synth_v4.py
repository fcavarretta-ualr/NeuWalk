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
    

def extract_neurites(morphology, section_type, expanded_filtering=False, to_delete=None):
  """
  Clone a morphology and extract top-level trees of selected section types.

  A selected section becomes a retained root when it has no parent or when
  its parent's section type is different.
  """
  
  root_sections = []

  # find the roots
  for root in morphology:

    # let's work on a copy
    for section in root.clone().subtree:
      # found a section of interest
      if section.section_type == section_type:

        # if the section is of interest but parent has a different section type
        # we should disconnect and treat as an independent tree
        if section.parent and section.parent.section_type != section_type:
          section.disconnect_from_parent()

        # now if it has no parent, it is a root
        if not section.parent:          
          root_sections.append(section)

        continue

    # by deleting they will be lost
    if to_delete and section.section_type in to_delete:
      section.disconnect_from_parent()

  # filter descendants from the root which are not of the same section
  if expanded_filtering:
    for root in root_sections:
        for section in root.subtree:
          if section.section_type != section_type:
            section.disconnect_from_parent()

  # merge with descendant if it has one child
  for root in root_sections:
    subtree = root.subtree

    while len(subtree):
      section = subtree.pop()

      if len(section.children) == 1:
        child = section.children[0]
        section._merge_with_descendant()
        if child in subtree:
          subtree.remove(child)
          
        
  # translate the root to the origin
  for root in root_sections:
      translate_subtree(root)
      
  return root_sections
    

    
def preprocess_morphology(morphology):
    """Split one loaded morphology into three independent cloned groups."""
    default = ["unknown", "axon", "apical_dendrite", "apical_secondary_dendrite", "apical_secondary_oblique"]
    
    basal_dendrites, apical_dendrites_basic, apical_dendrites_ext, apical_obliques = \
                     extract_neurites(morphology['morphology'], "basal_dendrite", to_delete=default, expanded_filtering=True), \
                     [], \
                     [], \
                     []
    return basal_dendrites, apical_dendrites_basic, apical_dendrites_ext, apical_obliques
    

def main():
    parser = argparse.ArgumentParser(
        description="Estimate event statistics and test EventSampler."
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cell-type", type=str, default="MITRAL")
    parser.add_argument("--soma-theta", type=float, default=0.)
    parser.add_argument("--soma-phi", type=float, default=0.)
    
    args = parser.parse_args()


    
    
    morphologies = load_morphologies(args.directory)

    stats = {}

    basal_dendrites, apical_dendrites, apical_dendrites_ext, apical_obliques = [], [], [], []
    for morphology in morphologies:
        _basal_dendrites, _apical_dendrites, _apical_dendrites_ext, _apical_obliques = \
                          preprocess_morphology(morphology)
        
        basal_dendrites.append(_basal_dendrites)
        apical_dendrites.append(_apical_dendrites)
        apical_dendrites_ext.append(_apical_dendrites_ext)
        apical_obliques.append(_apical_obliques)

    def chk_size(data):
      r = []
      for _r in data:
        r += _r
      return len(r) > 0


    if chk_size(basal_dendrites):
      stats['basal_dendrite'] = extract_statistics(
        basal_dendrites,
        bin_size=args.bin_size
        )
      
    if chk_size(apical_dendrites): 
      stats['apical_dendrite'] = extract_statistics(
        apical_dendrites,
        bin_size=args.bin_size
        )
      
    if chk_size(apical_dendrites_ext): 
      stats['apical_dendrite_ext'] = extract_statistics(
        apical_dendrites_ext,
        bin_size=args.bin_size
        )
      
    if chk_size(apical_obliques):
      stats['apical_oblique'] = extract_statistics(
        apical_obliques,
        bin_size=args.bin_size
        )



    #stats['apical_dendrite']['bifurcation_internal_density'] = stats['apical_dendrite_ext']['bifurcation_internal_density']
    
    #print('bifurcation_internal_density', stats['apical_dendrite']['bifurcation_internal_density'])

    
    profiles = {}
    for section_type, params in stats.items():
        if section_type != 'basal_dendrite':
          continue
        print('generating profile for ', section_type)

        del params['total_length']
        params['internal_event_sampler_parameters'] = None

        # set oblique parameters
##        oblique_present = section_type == "apical_dendrite" and "apical_oblique" in stats
####        if oblique_present:
####            tmp = stats['apical_oblique'].copy()
####            del tmp['total_length']
####
####            params['internal_event_sampler_parameters'] = tmp


        rng = Random(args.seed)

        topol_synthesizer = TopologySynthesizer(
            rng=rng,
            step_size=args.step_size,
            bin_size=args.bin_size,
            section_type=section_type,
            **params
        )

        topol_synthesizer.synthesize_progressive(
            n_std=1,
            max_attempts_per_window=5,
            max_total_attempts=1000,
            verbose=False,
        )
          
        #print('section_type:', section_type)
        #topol_synthesizer.describe()
        #print()

##        if oblique_present:
##          topol_synthesizer.activate_internal_branches()
##          topol_synthesizer.synthesize(1000)

        # attach the independently synthesized roots to a soma
        soma = NeuriteProfile(
            step_size=args.step_size,
            section_type="soma",
        )
        soma.children = list(topol_synthesizer.roots)
        
        for root in soma.children:
            root.parent = soma

        # store profile
        profiles[section_type] = soma
        
##    import assign_neurites as an
##    bin_size = 10
##    rng = misc.Random(args.seed)
##    ret = an.assign_neurites_to_bins(rng, int(profiles['apical_oblique'].sholl_plot(bin_size)[0]), bifurcation_internal_density)
##    
##
##
##    to_connect = []
##    iroot = 0
##    for n_internal, bin_index in ret:
##
##      for i in range(n_internal):
##        assign = an.neurites_by_distance_bin(profiles['apical_dendrite'].children, bin_index * bin_size, (bin_index + 1) * bin_size)
##
##        index = int(rng.random() * len(assign))
##
##        neurite = assign[index]['neurite']
##        interval = assign[index]['interval']
##        an.cut_neurite(rng, neurite, interval)
##
##        oblique_dendrite = profiles['apical_oblique'].children[iroot]
##        oblique_dendrite.internal_bifurcation = True
##        for _oblique_dendrite in oblique_dendrite._iter_sections():
##          _oblique_dendrite.order = neurite.order + 1
##        neurite.connect(oblique_dendrite, relation="child")
##        iroot += 1
        
        
##    b1 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (170., 0., 0.), (2.5, 2.5), (2.5, 2.5), 0.5, 1, strict=True)
##    b2 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 170.]), (310., 0., 0.), (2.5, 2.5), (50.0, 50.0), 0.5, 1, strict=True)
##    b3 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 480.]), (260., 0., 0.), (50.0, 50.0), (100.0, 100.0), 0.5, 1, strict=True)
##    b4 = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 740.]), (260., 0., 0.), (100.0, 100.0), (150.0, 150.0), 0.5, 1, strict=True)
##    b = b1 + b2 + b3 + b4
##
##    elongation_bias = []
##    elongation_bias.append((0.003, biases.get_elongation("sibling_repulsion", 25.0, -2)))
##    elongation_bias.append((0.003
##                            , biases.get_elongation("nonrelated_repulsion", 25.0, -2)))
##    elongation_bias.append((0.075, b))
##    elongation_bias.append((0.075, biases.get_elongation("plane_boundary", np.array([0., 0., 1000.]), (np.pi, 0.), 250, -2)))
    elongation_random_weight = 0.2
##    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 6)
##    bifurcation_internal_bias = biases.get_bifurcation("internal_branch", np.pi / 3)
##    
##    morph_synthesizer = MorphologySynthesizer(
##        root=profiles['basal_dendrite'],
##        rng=rng,
##        theta=0,
##        phi=0,
##        axis_direction=np.array([0.0, 0.0, 1.0]),
##        bifurcation_bias=bifurcation_bias,
##        bifurcation_internal_bias=bifurcation_internal_bias,
##        elongation_bias=elongation_bias,
##        elongation_random_weight=elongation_random_weight,
##        elongation_bias_weight=1
##    )
##    soma_apical = morph_synthesizer.synthesize()
##    print("bifurcation_count (Sim):", soma_apical.bifurcation_count)
##    print("sholl_plot:", soma_apical.sholl_plot(args.bin_size))
    
    ##elongation_bias = []
    #elongation_bias.append((0.001, biases.get_elongation("sibling_repulsion", 25.0, -2)))
    #elongation_bias.append((0.001, biases.get_elongation("nonrelated_repulsion", 10.0, -2)))
    #elongation_bias.append((0.001, biases.get_elongation("root_repulsion", None, None)))
    #oblique_synthesizer = morph_synthesizer.copy_with(elongation_bias=elongation_bias)
    #soma_apical = morph_synthesizer.synthesize()

    radii = np.array([2000, 1250, 1250], dtype=float)
    epl_depth = 300


    if args.cell_type == "MITRAL":
      inner_layer = radii - epl_depth
      theta = np.pi / 6
      b1 = biases.get_elongation("ellipsoid_boundary", radii, 150.0, -2, orientation='in')
      b2 = biases.get_elongation("ellipsoid_boundary", inner_layer, 75., -2, orientation='out')
    else:
      inner_layer = radii - epl_depth / 2
      theta = np.pi / 2
      b1 = biases.get_elongation("ellipsoid_boundary", radii, 1.0, -2, orientation='in')
      b2 = biases.get_elongation("ellipsoid_boundary", inner_layer, 50., -2, orientation='out')
    b = b1 + b2

    soma_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, args.soma_theta, args.soma_phi), inner_layer)
    glom_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, args.soma_theta, args.soma_phi), inner_layer)
    axis_direction = misc._normalize(misc.EllipsoidalCoordinates.to_cartesian((1, args.soma_theta, args.soma_phi), radii))
    
    elongation_bias = list()
    elongation_bias.append((0.001, biases.get_elongation("sibling_repulsion", 25.0, -2, space="ellipsoid", radii=radii)))
    elongation_bias.append((0.004, biases.get_elongation("nonrelated_repulsion", 10.0, -2, space="ellipsoid", radii=radii)))
    elongation_bias.append((0.03, b))

    bifurcation_bias = biases.get_bifurcation("cross_torsion", np.pi / 6, space="ellipsoid", radii=radii)
    
    morph_synthesizer = MorphologySynthesizer(
        root=profiles['basal_dendrite'],
        rng=rng,
        theta=theta,
        phi=(0, 2 * np.pi),
        axis_direction=axis_direction,
        origin=soma_position,
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight*0,
        centrifugal=True
    )
    soma_basal = morph_synthesizer.synthesize()
####    print("bifurcation_count (Sim):", soma_basal.bifurcation_count)
####    print("sholl_plot:", soma_basal.sholl_plot(args.bin_size))
####
####    print('children=', len(soma_basal.children), soma_basal.children)
####    for r in soma_basal.children:
####        r.disconnect(soma_basal)
####        r.connect(soma_apical, relation="parent")
    plot_morphology(soma_basal)

if __name__ == "__main__":
    main()
