#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from neuwalk.io import read_swc
from neuwalk.core.topology import SectionSynthesizer
from neuwalk.synthesis.topology.sampling import EventSampler
from neuwalk.random import Random


def load_statistics(directory, bin_size):
    """Extract population statistics from all SWC files in a directory."""
    files = sorted(Path(directory).glob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

    sholl_plots = []
    bifurcation_counts = []
    primary_counts = []

    for filename in files:
        roots = read_swc(filename)

        plots = [root.sholl_plot(bin_size) for root in roots]
        size = max(map(len, plots), default=0)
        total_sholl = np.zeros(size, dtype=float)

        for plot in plots:
            total_sholl[:len(plot)] += plot

        sholl_plots.append(total_sholl)
        bifurcation_counts.append(
            sum(root.bifurcation_count for root in roots)
        )
        primary_counts.append(
            sum(
                len(root.children)
                if root.label == "soma"
                else 1
                for root in roots
            )
        )

    size = max(map(len, sholl_plots))
    matrix = np.zeros((len(sholl_plots), size), dtype=float)

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
        "primary_count_range": {
            'min':int(np.min(primary_counts)),
            'max':int(np.max(primary_counts)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    stats = load_statistics(
        args.directory,
        args.bin_size,
    )

    rng = rng = Random(1234)

    sampler = EventSampler(
        rng=rng,
        step_size=args.step_size,
        bin_size=args.bin_size,
        sholl_plot=stats["sholl_plot"],
        bifurcation_count=stats["bifurcation_count"],
        primary_count_range=stats["primary_count_range"],
    )

    soma = SectionSynthesizer(
        step_size=args.step_size,
        label="soma",
    )

    primary_count = sampler.sample_primary_section_count()

    active = soma.create_primary_sections(
        number=primary_count,
        label="dendrite",
    )

    for step in range(args.max_steps):
        if not active:
            break

        events = [
            (section, sampler.sample_event(section))
            for section in active
        ]

        next_active = []

        for section, event in events:
            if event == "elongate":
                section.elongate()
                next_active.append(section)

            elif event == "bifurcate":
                next_active.extend(section.bifurcate())

            elif event == "bifurcate_internal":
                children = section.bifurcate_internal()
                next_active.append(children[0])

            elif event == "annihilate":
                section.annihilate()

            else:
                raise RuntimeError(f"Unknown event: {event!r}")

        active = next_active

    print("primary_count_range:", stats["primary_count_range"])
    print("sampled_primary_count:", primary_count)
    print("bifurcation_count:", soma.bifurcation_count())
    print("sholl_plot:", soma.sholl_plot(args.bin_size))


if __name__ == "__main__":
    main()
