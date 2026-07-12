#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from morphgenpy.io import read_swc


def iter_sections(roots, section_types=None):
    """Yield selected sections from all roots."""
    for root in roots:
        for section in root.subtree:
            if (
                section_types is None
                or section.section_type in section_types
            ):
                yield section


def iter_segments(roots, section_types=None):
    """Yield geometric segments belonging to selected sections."""
    for section in iter_sections(roots, section_types):
        points = np.asarray(section.points, dtype=float)

        for point_0, point_1 in zip(points[:-1], points[1:]):
            yield point_0, point_1

        if (
            section.parent is not None
            and len(section.parent.points)
            and len(points)
            and not np.allclose(
                section.parent.points[-1],
                points[0],
            )
        ):
            yield section.parent.points[-1], points[0]


def total_length(roots, section_types=None):
    """Return total length of selected sections."""
    return sum(
        np.linalg.norm(point_1 - point_0)
        for point_0, point_1 in iter_segments(
            roots,
            section_types,
        )
    )


def bifurcation_count(roots, section_types=None):
    """Return the number of bifurcations among selected sections."""
    count = 0

    for section in iter_sections(roots, section_types):
        selected_children = [
            child
            for child in section.children
            if (
                section_types is None
                or child.section_type in section_types
            )
        ]

        count += len(selected_children) >= 2

    return count


def sholl_plot(
    roots,
    bin_size,
    section_types=None,
    max_distance=None,
):
    """Return Sholl radii and intersection counts."""
    centers = [
        root.points[0]
        for root in roots
        if len(root.points)
    ]

    if not centers:
        return np.zeros(0), np.zeros(0, dtype=int)

    center = np.asarray(centers[0], dtype=float)
    segments = list(iter_segments(roots, section_types))

    if not segments:
        return np.zeros(0), np.zeros(0, dtype=int)

    if max_distance is None:
        max_distance = max(
            np.linalg.norm(point - center)
            for segment in segments
            for point in segment
        )

    radii = np.arange(
        0.0,
        max_distance + 0.5 * bin_size,
        bin_size,
    )
    counts = np.zeros(len(radii), dtype=int)

    for point_0, point_1 in segments:
        distance_0 = np.linalg.norm(point_0 - center)
        distance_1 = np.linalg.norm(point_1 - center)
        lower, upper = sorted((distance_0, distance_1))

        counts += (
            (radii > lower) & (radii <= upper)
        ).astype(int)

    return radii, counts


def find_swc_files(path, recursive=False):
    """Return SWC files found at a file or directory path."""
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() != ".swc":
            raise ValueError("The input file must have an .swc extension.")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    pattern = "**/*.swc" if recursive else "*.swc"
    files = sorted(path.glob(pattern))

    if not files:
        raise ValueError(f"No SWC files found in {path}.")

    return files


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Sholl profiles and distributions of bifurcation count "
            "and total length for SWC morphologies."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="SWC file or directory containing SWC files.",
    )
    parser.add_argument(
        "--section-types",
        type=int,
        nargs="+",
        default=None,
        help="Optional SWC section types to include, e.g. 3 4.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=10.0,
        help="Sholl shell spacing. Default: 10.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively.",
    )
    parser.add_argument(
        "--sholl-summary",
        action="store_true",
        help="Plot one mean Sholl curve with error bars.",
    )
    parser.add_argument(
        "--sholl-error",
        choices=("std", "sem"),
        default="std",
        help=(
            "Error bars for --sholl-summary: standard deviation "
            "or standard error. Default: std."
        ),
    )

    args = parser.parse_args()

    if args.bin_size <= 0:
        parser.error("--bin-size must be positive.")

    files = find_swc_files(
        args.path,
        recursive=args.recursive,
    )

    sholl_data = []
    branch_counts = []
    lengths = []

    for filename in files:
        roots = read_swc(filename)

        radii, counts = sholl_plot(
            roots,
            bin_size=args.bin_size,
            section_types=args.section_types,
        )

        sholl_data.append((filename.stem, radii, counts))
        branch_counts.append(
            bifurcation_count(
                roots,
                section_types=args.section_types,
            )
        )
        lengths.append(
            total_length(
                roots,
                section_types=args.section_types,
            )
        )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    if args.sholl_summary:
        max_bins = max(
            (len(counts) for _, _, counts in sholl_data),
            default=0,
        )

        if max_bins:
            matrix = np.full(
                (len(sholl_data), max_bins),
                np.nan,
                dtype=float,
            )

            for i, (_, _, counts) in enumerate(sholl_data):
                matrix[i, :len(counts)] = counts

            radii = np.arange(max_bins) * args.bin_size
            mean = np.nanmean(matrix, axis=0)
            sample_count = np.sum(~np.isnan(matrix), axis=0)

            if args.sholl_error == "std":
                error = np.nanstd(matrix, axis=0)
                error_label = "SD"
            else:
                error = np.divide(
                    np.nanstd(matrix, axis=0),
                    np.sqrt(sample_count),
                    out=np.zeros_like(mean),
                    where=sample_count > 0,
                )
                error_label = "SEM"

            axes[0].errorbar(
                radii,
                mean,
                yerr=error,
                marker="o",
                capsize=3,
                label=f"Mean ± {error_label}",
            )
            axes[0].legend(fontsize="small")
    else:
        for name, radii, counts in sholl_data:
            axes[0].plot(radii, counts, label=name)

        if len(sholl_data) <= 10:
            axes[0].legend(fontsize="small")

    axes[0].set_title("Sholl plot")
    axes[0].set_xlabel("Radius")
    axes[0].set_ylabel("Intersections")

    axes[1].hist(
        branch_counts,
        bins="auto",
        edgecolor="black",
    )
    axes[1].set_title("Bifurcation count")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Morphologies")

    axes[2].hist(
        lengths,
        bins="auto",
        edgecolor="black",
    )
    axes[2].set_title("Total length")
    axes[2].set_xlabel("Length")
    axes[2].set_ylabel("Morphologies")

    section_label = (
        "all section types"
        if args.section_types is None
        else f"section types: {args.section_types}"
    )

    figure.suptitle(
        f"{len(files)} morphology file(s), {section_label}"
    )
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
