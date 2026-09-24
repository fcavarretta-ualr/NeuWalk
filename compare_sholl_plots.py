"""
Plot a single Sholl comparison panel showing basal_dendrite (left,
negative x) and apical_dendrite (right, positive x) mirrored around
the soma at x=0 -- basal's bins are negated, never its crossing
counts -- comparing experimental (gray) and synthetic (black) neuron
morphologies, with a vertical dashed line at x=0 marking the soma.
Mean with error bars (std), line, and circle markers.

apical_dendrite is combined with apical_oblique (summed per neuron,
matched by filename) whenever a directory has any oblique data at
all -- checked independently for experimental and synthetic, so one
can include obliques while the other doesn't, if that's genuinely how
the two datasets differ. The legend reflects this per side, e.g.
"experimental apical (+oblique)" and/or "synthetic apical (+oblique)".

Sholl counts use Section's own sholl_plot(bin_size) method directly
(summed across roots, since a dendrite type can have several separate
primary roots once soma is deleted) rather than a reimplementation of
it.

Both directories are processed the same way extract_apc.py/extract_neo.py
do: via load_morphologies with each dendrite type's own delete_labels
(deleting every other dendrite type, soma, axon, and secondary variants
first, so what's left is only that type's own sections), with
soma_processing=False -- process_soma would otherwise insert a
spurious point at the start of every orphaned root.

Styling: top and right spines removed; the figure defaults to 7.27in
wide -- A4 portrait width (8.27in) minus a 0.5in margin on each side;
all text is bold and lines are thicker than matplotlib's own defaults.

Usage
-----
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR --bin-size 20
    python plot_sholl_comparison.py EXPERIMENTAL_DIR SYNTHETIC_DIR --output sholl.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LinearLocator

from neuwalk.analysis.morphologies import load_morphologies


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

    if total[-1] != 0:
        total = np.append(total, 0.0)

    return total


def load_sholl_plots_per_file(directory, subsection, bin_size):
    """
    Load every .swc file in directory, processed for the given
    dendrite type, and return a list of Sholl arrays (one per file
    that actually has at least one section of this type).
    """
    morphologies = load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[subsection], soma_processing=False)

    plots = []

    for roots in morphologies:
        sholl = label_sholl_plot(roots, bin_size)

        if sholl.sum() > 0:
            plots.append(sholl)

    return plots


def load_sholl_plots_per_file_with_names(directory, subsection, bin_size):
    """Like load_sholl_plots_per_file, but returns (filename, sholl_array) pairs instead of just the arrays."""
    morphologies = load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[subsection], return_file_names=True, soma_processing=False)

    plots = []

    for filename, roots in morphologies:
        sholl = label_sholl_plot(roots, bin_size)

        if sholl.sum() > 0:
            plots.append((filename, sholl))

    return plots


def load_apical_plots_per_file(directory, bin_size):
    """
    Load apical_dendrite Sholl plots per file, SUMMED with
    apical_oblique's own Sholl plot for the same file when a neuron
    has both (matched by filename, since a neuron can have one without
    the other -- the two are loaded and filtered independently, so
    simply zipping the two lists together isn't safe).

    Returns
    -------
    (list of numpy.ndarray, bool)
        The per-neuron combined Sholl arrays, and whether any oblique
        data was found in this directory at all (so the caller can
        label the result accordingly, e.g. "(+oblique)").
    """
    apical_by_file = dict(load_sholl_plots_per_file_with_names(directory, "apical_dendrite", bin_size))
    oblique_by_file = dict(load_sholl_plots_per_file_with_names(directory, "apical_oblique", bin_size))

    has_oblique = bool(oblique_by_file)

    plots = []

    for filename in set(apical_by_file) | set(oblique_by_file):
        apical_sholl = apical_by_file.get(filename, np.zeros(1))
        oblique_sholl = oblique_by_file.get(filename, np.zeros(1))

        max_len = max(len(apical_sholl), len(oblique_sholl))
        apical_sholl = np.pad(apical_sholl, (0, max_len - len(apical_sholl)))
        oblique_sholl = np.pad(oblique_sholl, (0, max_len - len(oblique_sholl)))

        plots.append(apical_sholl + oblique_sholl)

    return plots, has_oblique


def pad_to_common_length(arrays, target_length=None):
    """Pad a list of 1D arrays with zeros to a common length."""
    if target_length is None:
        target_length = max(len(a) for a in arrays)
    return np.array([np.pad(np.asarray(a, dtype=float), (0, target_length - len(a))) for a in arrays])


def plot_mirrored_panel(ax, apical_experimental, apical_synthetic, basal_experimental, basal_synthetic,
                         bin_size, marker_size, marker_type, linewidth, capsize, font_size,
                         experimental_apical_has_oblique, synthetic_apical_has_oblique):
    """
    Draw basal_dendrite and apical_dendrite on the same axes.

    If both are present, they mirror around x=0 (the soma): apical at
    positive x, basal at negative x -- only basal's BINS are negated,
    never its crossing counts. If only one is present, it's drawn at
    POSITIVE x instead (never negative), occupying the full extent,
    since there's nothing on the other side to mirror it against.
    """
    def draw(experimental_plots, synthetic_plots, sign, synthetic_color):
        nonlocal peak

        shared_length = max(
            [len(a) for a in experimental_plots] + [len(a) for a in synthetic_plots]
        ) if (experimental_plots or synthetic_plots) else 0

        if experimental_plots:
            array = pad_to_common_length(experimental_plots, shared_length)
            mean = array.mean(axis=0)
            std = array.std(axis=0)
            bins = sign * np.arange(len(mean)) * bin_size
            peak = max(peak, np.max(mean + std))

            ax.errorbar(bins, mean, yerr=std, color="gray", marker=marker_type, markersize=marker_size,
                        linestyle="-", linewidth=linewidth, elinewidth=linewidth, capsize=capsize, capthick=linewidth)

        if synthetic_plots:
            array = pad_to_common_length(synthetic_plots, shared_length)
            mean = array.mean(axis=0)
            std = array.std(axis=0)
            bins = sign * np.arange(len(mean)) * bin_size
            peak = max(peak, np.max(mean + std))

            ax.errorbar(bins, mean, yerr=std, color=synthetic_color, marker=marker_type, markersize=marker_size,
                        linestyle="-", linewidth=linewidth, elinewidth=linewidth, capsize=capsize, capthick=linewidth)

    peak = 0.0  # tallest (mean + std) drawn anywhere in the panel, tracked as we go

    apical_present = bool(apical_experimental or apical_synthetic)
    basal_present = bool(basal_experimental or basal_synthetic)

    if apical_present and basal_present:
        draw(apical_experimental, apical_synthetic, +1, "black")
        draw(basal_experimental, basal_synthetic, -1, "black")
    elif apical_present:
        draw(apical_experimental, apical_synthetic, +1, "black")
    elif basal_present:
        draw(basal_experimental, basal_synthetic, +1, "black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.tick_params(width=linewidth)

    # extra headroom above the tallest error bar (peak, tracked while
    # drawing), so the legend -- pinned to "upper right" rather than
    # matplotlib's own overlap-avoiding "best" -- has empty space to
    # sit in instead of covering the top of the data. 35% of the
    # panel's own data range is comfortably more than a 2-entry legend
    # box needs.
    ylim = ax.get_ylim()
    margin = 0.35 * (peak - ylim[0])
    ax.set_ylim(ylim[0], max(ylim[1], peak + margin))

    # vertical dashed line at x=0 marks the soma -- the shared
    # reference point both sides mirror around. Only meaningful when
    # both sides are actually shown; with only one dendrite type
    # present, there's no mirroring happening at x=0, so the line
    # would just sit at the left edge of the data rather than marking
    # anything real.
    if apical_present and basal_present:
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

    ax.set_xlabel("distance from soma (\u00b5m)", fontweight="bold", fontsize=font_size)
    ax.set_ylabel("crossings", fontweight="bold", fontsize=font_size)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(font_size)

    # legend entries only for colors/sides actually drawn. experimental
    # is split into "experimental apical"/"experimental basal" the
    # same way synthetic already is (even though both are gray),
    # rather than one shared "experimental" entry -- that single entry
    # would be ambiguous about which side any "(+oblique)" annotation
    # applies to once apical and basal are shown together.
    # exactly two entries -- one per color -- since apical and basal
    # now share the same color within each group (gray/black), so
    # separate per-side entries would just duplicate the same swatch.
    # "(+oblique)" still needs to be conveyed somehow, so it's appended
    # to the one entry for that group whenever ITS apical curve
    # includes oblique (oblique only ever relates to apical, so this
    # stays unambiguous despite not naming "apical" explicitly).
    handles = []

    if apical_experimental or basal_experimental:
        label = "experimental (+oblique)" if experimental_apical_has_oblique else "experimental"
        handles.append(Line2D([0], [0], color="gray", marker=marker_type, markersize=marker_size, linewidth=linewidth, label=label))

    if apical_synthetic or basal_synthetic:
        label = "synthetic (+oblique)" if synthetic_apical_has_oblique else "synthetic"
        handles.append(Line2D([0], [0], color="black", marker=marker_type, markersize=marker_size, linewidth=linewidth, label=label))

    legend = ax.legend(handles=handles, loc="upper right", fontsize=font_size)
    for text in legend.get_texts():
        text.set_fontweight("bold")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimental_dir", help="Directory of experimental .swc reconstructions.")
    parser.add_argument("synthetic_dir", help="Directory of synthetic/generated .swc reconstructions.")
    parser.add_argument("--bin-size", type=float, default=50.0, help="Sholl bin size (default: 50.0).")
    parser.add_argument("--marker-size", type=float, default=5.0, help="Size of the marker at each data point (default: 6.0).")
    parser.add_argument("--marker-type", default="o", help="Marker shape at each data point (default: 'o').")
    parser.add_argument("--linewidth", type=float, default=3.0, help="Line, error-bar, and spine thickness (default: 2.5).")
    parser.add_argument("--capsize", type=float, default=6.0, help="Width of the cap at each error bar's end (default: 8.0).")
    parser.add_argument("--font-size", type=float, default=9.0, help="Font size for axis labels, tick labels, and legend text (default: 6.0).")
    parser.add_argument("--exclude", nargs="+", choices=("apical_dendrite", "basal_dendrite"), default=[],
                         help="Exclude these dendrite type(s) entirely, even if data exists for them.")
    parser.add_argument("--exclude-oblique", action="store_true",
                         help="Never combine apical_dendrite with apical_oblique, even if oblique data is present -- apical stays plain apical_dendrite alone.")
    parser.add_argument("--fig-width", type=float, default=5.5, help="Total figure width in inches (default: 3.0).")
    parser.add_argument("--fig-height", type=float, default=2.25, help="Figure height in inches (default: 1.4).")
    parser.add_argument("--pad-inches", type=float, default=0.05, help="Whitespace kept around the plot when saving, in inches (default: 0.05). Used with bbox_inches='tight' to crop matplotlib's own default margins.")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Explicit x-axis range, overriding the automatic one.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Explicit y-axis range, overriding the automatic one (including the legend headroom margin).")
    parser.add_argument("--x-ticks", type=int, default=None, help="Exact number of tick marks on the x-axis (evenly spaced, including both ends), overriding matplotlib's automatic tick placement.")
    parser.add_argument("--y-ticks", type=int, default=None, help="Exact number of tick marks on the y-axis (evenly spaced, including both ends), overriding matplotlib's automatic tick placement.")
    parser.add_argument("--x-padding", type=float, default=0.0, help="Extra space (in x-axis data units) added on each side, between the actual plotted data and the axis edges (default: 0.0).")
    parser.add_argument("--y-padding", type=float, default=0.0, help="Extra space (in y-axis data units) added on each side, between the actual plotted data and the axis edges (default: 0.0).")
    parser.add_argument("--output", default=None, help="Save the figure to this path instead of showing it interactively.")
    args = parser.parse_args()

    experimental_apical_has_oblique = False
    synthetic_apical_has_oblique = False

    if "apical_dendrite" not in args.exclude:
        if args.exclude_oblique:
            apical_experimental = load_sholl_plots_per_file(args.experimental_dir, "apical_dendrite", args.bin_size)
            apical_synthetic = load_sholl_plots_per_file(args.synthetic_dir, "apical_dendrite", args.bin_size)
        else:
            apical_experimental, experimental_apical_has_oblique = load_apical_plots_per_file(args.experimental_dir, args.bin_size)
            apical_synthetic, synthetic_apical_has_oblique = load_apical_plots_per_file(args.synthetic_dir, args.bin_size)
    else:
        apical_experimental, apical_synthetic = [], []

    if "basal_dendrite" not in args.exclude:
        basal_experimental = load_sholl_plots_per_file(args.experimental_dir, "basal_dendrite", args.bin_size)
        basal_synthetic = load_sholl_plots_per_file(args.synthetic_dir, "basal_dendrite", args.bin_size)
    else:
        basal_experimental, basal_synthetic = [], []

    if not (apical_experimental or apical_synthetic or basal_experimental or basal_synthetic):
        print("No apical_dendrite or basal_dendrite data found in either directory.")
        return

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    plot_mirrored_panel(ax, apical_experimental, apical_synthetic, basal_experimental, basal_synthetic,
                         args.bin_size, args.marker_size, args.marker_type, args.linewidth, args.capsize, args.font_size,
                         experimental_apical_has_oblique, synthetic_apical_has_oblique)

    if args.xlim is not None:
        ax.set_xlim(*args.xlim)

    if args.ylim is not None:
        ax.set_ylim(*args.ylim)

    # applied after xlim/ylim (whether auto-computed or explicitly set
    # above), so it composes with either: expands whatever the current
    # range ends up being, rather than needing its own separate
    # data-range computation.
    if args.x_padding:
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - args.x_padding, xlim[1] + args.x_padding)

    if args.y_padding:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - args.y_padding, ylim[1] + args.y_padding)

    if args.x_ticks is not None:
        ax.xaxis.set_major_locator(LinearLocator(args.x_ticks))

    if args.y_ticks is not None:
        ax.yaxis.set_major_locator(LinearLocator(args.y_ticks))

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=1000, bbox_inches="tight", pad_inches=args.pad_inches)
        print(f"saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
