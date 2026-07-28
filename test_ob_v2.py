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

import json

def merge_profiles(profile_roots):
    # for non oblique, roots are attached to the soma          
    profile_soma = NeuriteProfile(1, section_type="soma")
    for root in profile_roots:
        root.connect(profile_soma, relation="parent")
    return profile_soma

def generate_apical(seed, step_size, soma_position, glom_position, axis_direction, elongation_random_weight, max_angle):
    topol_synthesizer = TopologySynthesizer(
        Random(seed),
        step_size=step_size,
        section_type="apical_dendrite",
        sholl_plot={'mean':[1,1], 'std':[0,0]},
        bin_size=np.linalg.norm(soma_position-glom_position),
        primary_count_range={"min":1, "max":1}
    )

    topol_synthesizer.synthesize()


    # initializa the synthesizer for apical dendrites
    apical_synthesizer = MorphologySynthesizer(
        root=merge_profiles(topol_synthesizer.roots),
        rng=misc.Random(seed=seed),
        theta=0,
        phi=0,
        origin=soma_position,
        axis_direction=axis_direction,
        elongation_bias=biases.get_elongation("attraction", None, None, glom_position),
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )

    # synthesize apical dendrites
    soma = apical_synthesizer.synthesize()

    # synthesize the tuft dendrites
    topol_synthesizer = TopologySynthesizer(
        Random(seed),
        step_size=step_size,
        section_type="apical_dendrite",
        sholl_plot={'mean':[5,10,20,40,80,80,40,20,10,5,0], 'std':[0,5,10,20,40,40,20,10,5,2.5,0]},
        bin_size=10,
        primary_count_range={"min":4, "max":6}
    )

    topol_synthesizer.synthesize()


    center = apical_synthesizer.soma.children[0].points[-1] + axis_direction * 50.0

    # initializa the synthesizer for apical dendrites
    tuft_synthesizer = MorphologySynthesizer(
        root=merge_profiles(topol_synthesizer.roots),
        rng=misc.Random(seed=seed),
        theta=0,
        phi=0,
        origin=apical_synthesizer.soma.children[0].points[-1],
        axis_direction=axis_direction,
        elongation_bias=biases.get_elongation("ellipsoid_boundary", np.zeros(3, dtype=float) + 50.0, 1, -1, orientation='in', center=center),
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )

    # synthesize apical dendrites
    tuft_origin = tuft_synthesizer.synthesize()
    for ch in tuft_origin.children:
        ch.disconnect_from_parent()
        ch.connect(soma, relation="parent")
    return soma

    

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
    parser.add_argument("--theta", type=float, default=0)
    parser.add_argument("--phi", type=float, default=0)
    parser.add_argument("--cell-type", type=str, default="MITRAL")
    args = parser.parse_args()


    
##    if args.cell_type == "MITRAL":
##        soma_layer = radii - epl_depth
##        primary_theta = np.pi / 6
##        spatial_bias = biases.get_elongation("ellipsoid_boundary", radii, epl_depth / 2, -1, orientation='in') + \
##                       biases.get_elongation("ellipsoid_boundary", soma_layer, 75., -1, orientation='out')
##    elif args.cell_type == "TUFTED":
##        soma_layer = radii - epl_depth / 2
##        primary_theta = np.pi / 2
##        spatial_bias = biases.get_elongation("ellipsoid_boundary", radii, 1.0, -1, orientation='in') + \
##                       biases.get_elongation("ellipsoid_boundary", soma_layer, 50., -1, orientation='out')
##    else:
##        raise ValueError("Unknown cell type.")
##
##    
##
##    soma_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, args.theta, args.phi), soma_layer)
##    glom_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, args.theta, args.phi), radii)
##    axis_direction = misc.to_unit_vector(misc.EllipsoidalCoordinates.to_cartesian((1, args.theta, args.phi), radii))
##
##    print(f"soma_position={soma_position},\tglom_position={glom_position}")
##    
##    elongation_random_weight = 1
##    max_angle = np.pi / 2
##    
##    soma = generate_apical(args.seed, args.step_size, soma_position, glom_position, axis_direction, elongation_random_weight, max_angle)
    
    # extract statistics for basal, apical, and oblique dendrites
    # param contains the sections to be discarded
    discarded_sections = {
      'basal_dendrite':["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "apical_dendrite", "axon"]
      }

    # stat contains the statistics
    all_params = {}
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
      
      all_params[section_type] = params.copy()
      all_params[section_type]['bin_size'] = args.bin_size
      
      with open("parameters.json", "w") as file:
          json.dump(params, file, indent=4, default=lambda value: value.tolist())

      quit()
              
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


    all_params.pop("apical_dendrite", None)    
    
    with open("parameters.json", "w") as file:
      json.dump(all_params, file, indent=4, default=lambda value: value.tolist())
    quit()   

    # create self-avoidance bias
    r1 = biases.get_elongation("sibling_repulsion", 25.0, -2) #, space="ellipsoid", radii=radii)
    r2 = biases.get_elongation("nonrelated_repulsion", 25.0, -2) #, space="ellipsoid", radii=radii)
    rsoma = biases.get_elongation("root_repulsion", 750, -2) #, space="ellipsoid", radii=radii)
    r = r1 + r2
    
    elongation_bias = [ (0.25, spatial_bias), (0.005, r), (0.2, rsoma) ]
    
    bifurcation_bias = biases.get_bifurcation("cross_torsion", np.pi / 3, space="ellipsoid", radii=radii)

    # initializa the synthesizer for apical dendrites
    basal_synthesizer = MorphologySynthesizer(
        root=merge_profiles(profiles['basal_dendrite']),
        rng=misc.Random(seed=args.seed),
        theta=primary_theta,
        phi=(0, 2 * np.pi),
        origin=soma_position,
        axis_direction=axis_direction,
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        elongation_random_weight=elongation_random_weight,
        max_angle=max_angle
    )

    # synthesize apical dendrites
    basal_synthesizer.synthesize()
     

    
    # unified basal and apical dendrites
    for r in basal_synthesizer.soma.children:
        r.disconnect_from_parent()
        r.connect(soma, relation="parent")


    # plot the morphology
    plot_morphology(soma)

if __name__ == "__main__":
    main()
