#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from neuwalk.io import read_swc
from neuwalk.core.topology import SectionSynthesizer
from neuwalk.synthesis.topology.sampling import EventSampler
from neuwalk.random import Random




def load_statistics(directory, bin_size):
    files = sorted(Path(directory).glob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

    sholl_plots = []
    bifurcation_counts = []

    for filename in files:
        roots = read_swc(filename)

        plots = [
            root.sholl_plot(bin_size)
            for root in roots
        ]

        size = max(map(len, plots))
        total = np.zeros(size)

        for plot in plots:
            total[:len(plot)] += plot

        sholl_plots.append(total)
        bifurcation_counts.append(
            sum(root.bifurcation_count for root in roots)
        )

    size = max(map(len, sholl_plots))
    matrix = np.zeros((len(sholl_plots), size))

    for i, plot in enumerate(sholl_plots):
        matrix[i, :len(plot)] = plot
        
    return {
        "sholl_plot": {
            "mean": matrix.mean(axis=0),
            "std": matrix.std(axis=0),
        },
        "bifurcation_count": {
            "mean": float(np.mean(bifurcation_counts)),
            "std": float(np.std(bifurcation_counts)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    stats = load_statistics(args.directory, args.bin_size)

    rng = Random(1234)
    section = SectionSynthesizer(step_size=args.step_size, label="apical_dendrite")
    sampler = EventSampler(
        rng=rng,
        step_size=args.step_size,
        bin_size=args.bin_size,
        sholl_plot=stats["sholl_plot"],
        bifurcation_count=stats["bifurcation_count"],
    )



    for i in range(args.samples):
        print(
            f"{i:3d}  "
            f"{sampler.sample_event(section)}"
        )


if __name__ == "__main__":
    main()
