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
from morphgenpy.analysis import load_statistics
from morphgenpy.misc import Random




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

    # extract information about apical dendrites
    stats = load_statistics(args.directory, args.bin_size, root_section_types=["apical_dendrite"])


    rng = Random(args.seed)

    topol_synthesizer = TopologySynthesizer(
        rng=rng,
        step_size=args.step_size,
        bin_size=args.bin_size,
        sholl_plot=stats["sholl_plot"],
        bifurcation_count=stats["bifurcation_count"],
        primary_count_range=stats["primary_count_range"],
        no_bifurcation_bins=stats["no_bifurcation_bins"],
        no_annihilation_bins=stats["no_annihilation_bins"],
        section_type="apical_dendrite"
    )

    topol_synthesizer.synthesize_progressive(
        n_std=1.0,
        max_attempts_per_window=5,
        max_total_attempts=1000,
        verbose=False,
    )

    
##    print("primary_count_range:", stats["primary_count_range"])
##    print("sampled_primary_count:", len(topol_synthesizer.soma.children))
##    print(
##        "internal_density:",
##        stats["bifurcation_internal_density"],
##    )
##    print(
##        "no_bifurcation_bins:",
##        stats["no_bifurcation_bins"],
##    )
##    print(
##        "no_annihilation_bins:",
##        stats["no_annihilation_bins"],
##    )
##    print("bifurcation_count (Exp):", stats["bifurcation_count"])
##    print("bifurcation_count (Sim):", topol_synthesizer.soma.bifurcation_count())
##    print("sholl_plot:", topol_synthesizer.soma.sholl_plot(args.bin_size))
##
##    elongation_bias = []
##    elongation_bias.append((0.002, biases.get_elongation("sibling_repulsion", 25.0, -2)))
##    elongation_bias.append((0.002, biases.get_elongation("nonrelated_repulsion", 10.0, -2)))
##    elongation_bias.append((0.1, biases.get_elongation("truncated_cone_boundary", np.zeros(3), (800., 0., 0.), (2.5, 1.25), (100.0, 25.0), 0.5, 2)))
##    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 6)
##    
##    morph_synthesizer = MorphologySynthesizer(
##        root=topol_synthesizer.soma,
##        rng=rng,
##        theta={1:0, 'default':np.pi/6},
##        phi={1:0, 'default':(0, 2*np.pi)},
##        axis_direction=np.array([0.0, 0.0, 1.0]),
##        bifurcation_bias=bifurcation_bias,
##        elongation_bias=elongation_bias,
##        elongation_random_weight=elongation_random_weight,
##        elongation_bias_weight=elongation_bias_weight,
##    )
##    soma_apical = morph_synthesizer.synthesize()
##    print("bifurcation_count (Sim):", soma_apical.bifurcation_count)
##    print("sholl_plot:", soma_apical.sholl_plot(args.bin_size))
##    
##    stats = load_statistics(
##        args.directory,
##        args.bin_size,
##        root_section_types=["basal_dendrite"]
##    )
##
##
##    rng = Random(args.seed * 2)
##
##    topol_synthesizer = TopologySynthesizer(
##        rng=rng,
##        step_size=args.step_size,
##        bin_size=args.bin_size,
##        sholl_plot=stats["sholl_plot"],
##        bifurcation_count=stats["bifurcation_count"],
##        primary_count_range=stats["primary_count_range"],
##        bifurcation_internal_density=stats["bifurcation_internal_density"],
##        no_bifurcation_bins=stats["no_bifurcation_bins"],
##        no_annihilation_bins=stats["no_annihilation_bins"],
##        section_type="basal_dendrite"
##    )
##
##    topol_synthesizer.synthesize_progressive(
##        n_std=1.0,
##        max_attempts_per_window=5,
##        max_total_attempts=1000,
##        verbose=False,
##    )
##    
##    print("primary_count_range:", stats["primary_count_range"])
##    print("sampled_primary_count:", len(topol_synthesizer.soma.children))
##    print(
##        "internal_density:",
##        stats["bifurcation_internal_density"],
##    )
##    print(
##        "no_bifurcation_bins:",
##        stats["no_bifurcation_bins"],
##    )
##    print(
##        "no_annihilation_bins:",
##        stats["no_annihilation_bins"],
##    )
##    print("bifurcation_count (Exp):", stats["bifurcation_count"])
##    print("bifurcation_count (Sim):", topol_synthesizer.soma.bifurcation_count())
##    print("sholl_plot:", topol_synthesizer.soma.sholl_plot(args.bin_size))
##    print('children=', len(topol_synthesizer.soma.children))
##
##    elongation_bias = list()
##    elongation_bias.append((0.001, biases.get_elongation("sibling_repulsion", 25.0, -2)))
##    elongation_bias.append((0.001, biases.get_elongation("nonrelated_repulsion", 10.0, -2)))
##    elongation_bias.append((0.1, biases.get_elongation("truncated_cone_boundary", np.zeros(3), (800., np.pi, 0.), (2.5, 1.25), (100.0, 25.0), 0.5, 2)))
##    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 6)
##    
##    morph_synthesizer = MorphologySynthesizer(
##        root=topol_synthesizer.soma,
##        rng=rng,
##        theta=np.pi / 6,
##        phi=(0, 2 * np.pi),
##        axis_direction=np.array([0.0, 0.0, -1.0]),
##        bifurcation_bias=bifurcation_bias,
##        elongation_bias=elongation_bias,
##        elongation_random_weight=elongation_random_weight,
##        elongation_bias_weight=elongation_bias_weight,
##    )
##    soma_basal = morph_synthesizer.synthesize()
##    print("bifurcation_count (Sim):", soma_basal.bifurcation_count)
##    print("sholl_plot:", soma_basal.sholl_plot(args.bin_size))
##
##    print('children=', len(soma_basal.children), soma_basal.children)
##    for r in soma_basal.children:
##        r.disconnect(soma_basal)
##        r.connect(soma_apical, relation="parent")
##    plot_morphology(soma_apical)

if __name__ == "__main__":
    main()
