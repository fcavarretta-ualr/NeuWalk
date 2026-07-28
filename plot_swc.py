#!/usr/bin/env python3

import argparse
from pathlib import Path

from neuwalk.io import read_swc
from neuwalk.visualization import plot_morphology


def main():
    parser = argparse.ArgumentParser(
        description="Load and plot a morphology from an SWC file."
    )
    parser.add_argument(
        "filename",
        type=Path,
        help="Path to the SWC morphology file.",
    )
    parser.add_argument(
        "--color-sections",
        action="store_true",
        help="Color sections according to their SWC section type.",
    )

    args = parser.parse_args()

    if not args.filename.is_file():
        parser.error(f"File not found: {args.filename}")

    roots = read_swc(args.filename)

    section_colors = None

    if args.color_sections:
        section_colors = {
            1: "black",   # soma
            2: "red",     # axon
            3: "blue",    # basal dendrite
            4: "green",   # apical dendrite
        }

    for root in roots:
        plot_morphology(
            root,
            section_colors=section_colors,
            show=False,
        )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
