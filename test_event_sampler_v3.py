#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from neuwalk.io import read_swc
from neuwalk.profiles import SectionProfile
from neuwalk.sampling import EventSampler


def _pad(array, size):
    """Pad a one-dimensional array with zeros."""
    result = np.zeros(size, dtype=float)
    result[:len(array)] = array
    return result


def load_statistics(directory, bin_size):
    """Extract event and Sholl statistics from all SWC files."""
    files = sorted(Path(directory).glob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

    records = []
    primary_counts = []

    for filename in files:
        roots = read_swc(filename)

        sholl = []
        bifurcations = []
        annihilations = []
        internal_bifurcations = []

        for root in roots:
            root_sholl = root.sholl_plot(bin_size)
            bif, ann, internal = root._event_counts(bin_size)

            sholl.append(root_sholl)
            bifurcations.append(bif)
            annihilations.append(ann)
            internal_bifurcations.append(internal)

        sholl_size = max(map(len, sholl), default=1)
        event_size = max(sholl_size - 1, 0)

        total_sholl = sum(
            (_pad(values, sholl_size) for values in sholl),
            start=np.zeros(sholl_size),
        )
        total_bif = sum(
            (_pad(values, event_size) for values in bifurcations),
            start=np.zeros(event_size),
        )
        total_ann = sum(
            (_pad(values, event_size) for values in annihilations),
            start=np.zeros(event_size),
        )
        total_internal = sum(
            (_pad(values, event_size) for values in internal_bifurcations),
            start=np.zeros(event_size),
        )

        records.append(
            (
                total_sholl,
                total_bif,
                total_ann,
                total_internal,
            )
        )

        primary_counts.append(
            int(total_sholl[0])
        )

    sholl_size = max(len(record[0]) for record in records)
    event_size = sholl_size - 1

    sholl_matrix = np.vstack([
        _pad(record[0], sholl_size)
        for record in records
    ])
    bif_matrix = np.vstack([
        _pad(record[1], event_size)
        for record in records
    ])
    ann_matrix = np.vstack([
        _pad(record[2], event_size)
        for record in records
    ])
    internal_matrix = np.vstack([
        _pad(record[3], event_size)
        for record in records
    ])

    mean_sholl = sholl_matrix.mean(axis=0)

    # Mean internal events divided by mean branch length represented in each
    # spatial bin. The final Sholl entry has no corresponding event bin.
    exposure = mean_sholl[:-1] * bin_size
    internal_density = np.divide(
        internal_matrix.mean(axis=0),
        exposure,
        out=np.zeros(event_size, dtype=float),
        where=exposure > 0,
    )

    no_bifurcation = bif_matrix.sum(axis=0) == 0
    no_annihilation = ann_matrix.sum(axis=0) == 0

    bifurcation_counts = bif_matrix.sum(axis=1)

    return {
        "sholl_plot": {
            "mean": mean_sholl,
            "std": sholl_matrix.std(axis=0),
        },
        "bifurcation_count": {
            "mean": float(bifurcation_counts.mean()),
            "std": float(bifurcation_counts.std()),
        },
        "primary_count_range": {
            'min':int(np.min(primary_counts)),
            'max':int(np.max(primary_counts)),
        },
        "bifurcation_internal_density": internal_density,
        "no_bifurcation_bins": no_bifurcation.tolist(),
        "no_annihilation_bins": no_annihilation.tolist(),
    }


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

    stats = load_statistics(
        args.directory,
        args.bin_size,
    )

    rng = np.random.default_rng(args.seed)

    sampler = EventSampler(
        rng=rng,
        step_size=args.step_size,
        bin_size=args.bin_size,
        sholl_plot=stats["sholl_plot"],
        bifurcation_count=stats["bifurcation_count"],
        primary_count_range=stats["primary_count_range"],
        bifurcation_internal_density=(
            stats["bifurcation_internal_density"]
        ),
        no_bifurcation_bins=stats["no_bifurcation_bins"],
        no_annihilation_bins=stats["no_annihilation_bins"],
    )

    soma = SectionProfile(
        step_size=args.step_size,
        label="soma",
    )

    primary_count = sampler.sample_primary_section_count()

    active = soma.create_primary_sections(
        number=primary_count,
        label="basal_dendrite",
    )

    for _ in range(args.max_steps):
        if not active:
            break

        sampled = [
            (section, sampler.sample_event(section))
            for section in active
        ]

        next_active = []

        for section, event in sampled:
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
                raise RuntimeError(
                    f"Unknown event: {event!r}."
                )

        active = next_active

    print("primary_count_range:", stats["primary_count_range"])
    print("sampled_primary_count:", primary_count)
    print(
        "internal_density:",
        stats["bifurcation_internal_density"],
    )
    print(
        "no_bifurcation_bins:",
        stats["no_bifurcation_bins"],
    )
    print(
        "no_annihilation_bins:",
        stats["no_annihilation_bins"],
    )
    print("bifurcation_count:", soma.bifurcation_count())
    print("sholl_plot:", soma.sholl_plot(args.bin_size))


if __name__ == "__main__":
    main()
