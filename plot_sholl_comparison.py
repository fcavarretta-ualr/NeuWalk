"""
Plot Sholl plots -- mean with error bars (std), line, and circle
markers -- one panel per dendrite type present in either directory,
comparing experimental (gray) and synthetic (black) neuron
morphologies.

Sholl counts use Section's own sholl_plot(bin_size) method directly
(summed across roots, since a dendrite type can have several separate
primary roots once soma is deleted) rather than a reimplementation of
it.

Both directories are processed the same way extract_apc.py/extract_neo.py
do: via load_morphologies with each dendrite type's own delete_labels
(deleting every other dendrite type, soma, axon, and secondary variants
first, so what's left is only that type's own sections), with
soma_processing=False -- process_soma would otherwise insert a spurious
point at the start of every orphaned root, harmless for apical/basal
(which attach near the real soma) but substantially distorting for
apical_oblique (which attaches at scattered points along the trunk).

If loading or plotting a given dendrite type raises an error, that
type is skipped (with a warning printed) rather than crashing the
whole script -- the remaining dendrite types are still loaded and
plotted normally.

Usage
-----
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR --bin-size 20
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR --dendrite-types apical_dendrite basal_dendrite
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR --fig-width 8.27 --output sholl.png
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR --output sholl.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from neuwalk.analysis.morphologies import load_morphologies


SUBSECTIONS = ("apical_dendrite", "basal_dendrite", "apical_oblique")

# Matches extract_apc.py/extract_neo.py's own discarded_sections exactly:
# for a given dendrite type's own statistics, every section belonging
# to a DIFFERENT dendritic lineage (plus soma, axon, unknown, and
# secondary oblique/dendrite variants) is deleted before measuring.
LABEL_DELETE_SETS = {
    "basal_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite", "axon"],
    "apical_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
    "apical_oblique": ["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
}


def label_sholl_plot(roots, bin_size):
    """
    Sholl crossing counts for a (possibly multi-root) dendrite type,
    using Section's own sholl_plot(bin_size) rather than a
    reimplementation of it.

    Section.sholl_plot operates on one Section's own subtree, so with
    soma_processing=False and soma itself deleted, this dendrite type's
    sections form several separate, unconnected roots (not one shared
    tree under a common soma) -- root.sholl_plot(bin_size) is called
    once per root, and the results (zero-padded to a common length)
    are SUMMED, since crossings from different roots at the same
    distance add together.

    Returns
    -------
    numpy.ndarray
        Length is one more than the furthest bin any root reaches (plus
        one more if that bin wasn't already zero -- see below); no
        roots at all returns an all-zero array of length 1.
    """
    if not roots:
        return np.zeros(1, dtype=int)

    per_root = [np.asarray(root.sholl_plot(bin_size), dtype=float) for root in roots]
    max_len = max(len(a) for a in per_root)
    padded = [np.pad(a, (0, max_len - len(a))) for a in per_root]

    total = np.sum(padded, axis=0)

    # sholl_plot's default sizing stops exactly at the bin containing
    # the furthest actual point, which is essentially always nonzero
    # by construction (that's where the tree actually reaches) -- left
    # as-is, the plotted curve looks abruptly cut off mid-slope rather
    # than tapering to zero. If the last bin isn't already zero, add
    # one more, zero-valued bin, since there are genuinely zero
    # crossings just beyond the tree's actual reach.
    if total[-1] != 0:
        total = np.append(total, 0.0)

    return total


def load_sholl_plots_per_file(directory, subsection, bin_size):
    """
    Load every .swc file in directory, processed for the given
    dendrite type, and return a list of Sholl arrays (one per file
    that actually has at least one section of this type -- a file
    with none contributes nothing, rather than an all-zero array that
    would misrepresent its absence as a real, zero-crossing
    measurement).
    """
    morphologies = load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[subsection], soma_processing=False)

    plots = []

    for roots in morphologies:
        sholl = label_sholl_plot(roots, bin_size)

        if sholl.sum() > 0:
            plots.append(sholl)

    return plots


def pad_to_common_length(arrays, target_length=None):
    """
    Pad a list of 1D arrays with zeros to a common length: target_length
    if given, otherwise the max length within arrays itself.
    """
    if target_length is None:
        target_length = max(len(a) for a in arrays)
    return np.array([np.pad(np.asarray(a, dtype=float), (0, target_length - len(a))) for a in arrays])


def plot_panel(ax, subsection, experimental_plots, synthetic_plots, bin_size, marker_size, marker_type):
    # both lines must have the SAME number of bins -- padding each
    # group to its own max length independently (as pad_to_common_length
    # does on its own) can still leave the two groups at different
    # lengths from EACH OTHER whenever one group's neurons reach
    # further than the other's, so the shared target length is the max
    # across both groups together, not each group's own max.
    shared_length = max(
        [len(a) for a in experimental_plots] + [len(a) for a in synthetic_plots]
    ) if (experimental_plots or synthetic_plots) else 0

    if experimental_plots:
        array = pad_to_common_length(experimental_plots, shared_length)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        bins = np.arange(len(mean)) * bin_size

        ax.errorbar(bins, mean, yerr=std, color="gray", marker=marker_type, markersize=marker_size, linestyle="-",
                    capsize=3, label=f"experimental (n={len(experimental_plots)})")

    if synthetic_plots:
        array = pad_to_common_length(synthetic_plots, shared_length)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        bins = np.arange(len(mean)) * bin_size

        ax.errorbar(bins, mean, yerr=std, color="black", marker=marker_type, markersize=marker_size, linestyle="-",
                    capsize=3, label=f"synthetic (n={len(synthetic_plots)})")

    ax.set_title(subsection)
    ax.set_xlabel("distance from soma")
    ax.set_ylabel("crossings")
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimental_dir", help="Directory of experimental .swc reconstructions.")
    parser.add_argument("synthetic_dir", help="Directory of synthetic/generated .swc reconstructions.")
    parser.add_argument("--bin-size", type=float, default=50.0, help="Sholl bin size (default: 50.0).")
    parser.add_argument("--marker-size", type=float, default=6.0, help="Size of the marker at each data point (default: 6.0).")
    parser.add_argument("--marker-type", default="o", help="Marker shape at each data point, using matplotlib marker codes -- e.g. 'o' circle, 's' square, '^' triangle, 'D' diamond (default: 'o').")
    parser.add_argument("--dendrite-types", nargs="+", choices=SUBSECTIONS, default=list(SUBSECTIONS),
                         help=f"Which dendrite type(s) to plot, one or more of {SUBSECTIONS} (default: all present).")
    parser.add_argument("--fig-width", type=float, default=None, help="Total figure width in inches (default: 6 * number of panels shown). A4 portrait width is 8.27.")
    parser.add_argument("--fig-height", type=float, default=5.0, help="Figure height in inches (default: 5.0). A4 portrait height is 11.69.")
    parser.add_argument("--output", default=None, help="Save the figure to this path instead of showing it interactively.")
    args = parser.parse_args()

    panels = []

    for subsection in args.dendrite_types:
        try:
            experimental_plots = load_sholl_plots_per_file(args.experimental_dir, subsection, args.bin_size)
            synthetic_plots = load_sholl_plots_per_file(args.synthetic_dir, subsection, args.bin_size)
        except Exception as error:
            print(f"warning: skipping {subsection} -- failed to load: {error}")
            continue

        if not experimental_plots and not synthetic_plots:
            continue

        panels.append((subsection, experimental_plots, synthetic_plots))

    if not panels:
        print(f"No data found for any of {args.dendrite_types} in either directory.")
        return

    fig_width = args.fig_width if args.fig_width is not None else 6 * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, args.fig_height), squeeze=False)
    axes = axes[0]

    for ax, (subsection, experimental_plots, synthetic_plots) in zip(axes, panels):
        try:
            plot_panel(ax, subsection, experimental_plots, synthetic_plots, args.bin_size, args.marker_size, args.marker_type)
        except Exception as error:
            print(f"warning: skipping {subsection} -- failed to plot: {error}")
            ax.set_visible(False)

    fig.suptitle("Sholl plot: experimental (gray) vs. synthetic (black), mean \u00b1 std")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
