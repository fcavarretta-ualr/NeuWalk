#!/usr/bin/env python3

import argparse
from pathlib import Path

from morphgenpy.io import read_swc
from morphgenpy.visualization import plot_morphology


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
            "soma": "black",   # soma
            "axon": "red",     # axon
            "basal_dendrite": "blue",    # basal dendrite
            "apical_dendrite": "green",   # apical dendrite
        }
        
    plot_morphology(
        roots,
        section_colors=section_colors,
        show=False,
    )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
