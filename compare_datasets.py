"""
Compare two SWC morphology datasets, in two different directories, using
bootstrap resampling.

Datasets are loaded the same way extract_apc.py does: via
neuwalk.analysis.morphologies.load_morphologies, which runs
process_morphology on every file (deleting sections outside the label
being analyzed, merging the resulting single-child sections together,
pruning short terminal leaves, and recentering coordinates). Since
different labels need different sections deleted, each label's data is
loaded SEPARATELY (re-reading from disk per label, exactly as
extract_apc.py's own per-label loop does) rather than loading once and
reusing the same tree across labels.

Two comparisons are performed, both for apical_dendrite and
basal_dendrite separately:

1. Bifurcation count and total length, comparing both the mean and the
   variance, one value per neuron.
2. Sholl-plot crossing counts, bin by bin, comparing both the mean and
   the variance across every neuron in each dataset.

Each comparison reports, per metric: each dataset's value, the observed
difference (A - B), a bootstrap confidence interval for that
difference, and a bootstrap p-value for whether it differs from zero.

Usage
-----
    python compare_datasets.py DIR_A DIR_B
    python compare_datasets.py DIR_A DIR_B --bin-size 20 --n-resamples 20000
"""

import argparse

import numpy as np

from neuwalk.analysis.morphologies import load_morphologies
import bootstrap_test as bs

LABELS = ("apical_dendrite", "basal_dendrite", "apical_oblique")

# Matches extract_apc.py's discarded_sections exactly: for each label's
# own statistics, every section belonging to a DIFFERENT dendritic
# lineage is deleted before counting -- e.g. computing apical_dendrite's
# stats deletes basal_dendrite and every oblique-related label, so what
# remains is only the apical trunk (obliques merged/pruned away, not
# just excluded from the count).
LABEL_DELETE_SETS = {
    "basal_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite", "axon"],
    "apical_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
    "apical_oblique": ["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
}


def load_dataset(directory, label):
    """
    Load every .swc file in directory for the given label, processed the
    same way extract_apc.py processes it (via load_morphologies with
    that label's own delete_labels). Returns a list of neurons, each
    itself a list of root sections -- there is no single soma root
    here, since "soma" is deleted for every label (see
    LABEL_DELETE_SETS), leaving each primary section of the target
    label as its own, separate root.
    """
    return load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[label])



def label_bifurcation_count(roots):
    """
    Bifurcation count restricted to sections with the given label,
    matching extract_statistics' definition exactly: a section with 2
    children only counts here if BOTH children share the section's own
    label. A section whose children have DIFFERENT labels from each
    other (e.g. one continuing the same label, one starting an oblique)
    is an "internal bifurcation" in extract_statistics' terms and is
    excluded from bifurcation_count there -- so it must be excluded
    here too, or this count won't match extract_statistics' for any
    preset that grafts a different-labeled branch (e.g. neocortex's
    apical_oblique).
    """
    return sum(section.bifurcation_count for section in roots)


def label_total_length(roots):
    """Total length restricted to sections with the given label."""
    return sum(section.total_length for section in roots)


def bootstrap_compare(a, b, statistic=np.mean, n_resamples=10000, ci=0.95, rng=None):
    """
    Compare two independent samples via bootstrap resampling of the
    difference in a chosen statistic (statistic(a) - statistic(b)).

    Parameters
    ----------
    a, b : array-like
        Independent samples (e.g. one value per neuron), from dataset A
        and dataset B respectively.
    statistic : callable, default numpy.mean
        Statistic to compare, e.g. numpy.mean or numpy.var. Called as
        statistic(array) -> float.
    n_resamples : int, default 10000
        Number of bootstrap resamples.
    ci : float, default 0.95
        Confidence level for the reported interval.
    rng : numpy.random.Generator, optional
        Random number generator. A fresh, unseeded one is created if not
        given.

    Returns
    -------
    dict
        value_a, value_b, observed_diff, ci_low, ci_high, p_value.
    """
    if rng is None:
        rng = np.random.default_rng()

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError("Both samples must be nonempty.")

    value_a = statistic(a)
    value_b = statistic(b)
    observed_diff = value_a - value_b

    boot_diffs = np.empty(n_resamples)

    for i in range(n_resamples):
        resample_a = rng.choice(a, size=a.size, replace=True)
        resample_b = rng.choice(b, size=b.size, replace=True)
        boot_diffs[i] = statistic(resample_a) - statistic(resample_b)

    alpha = 1.0 - ci
    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided bootstrap p-value: how often the bootstrap difference
    # falls on the opposite side of zero from the observed direction,
    # doubled for two-sidedness, capped at 1.
    p_low = np.mean(boot_diffs <= 0.0)
    p_high = np.mean(boot_diffs >= 0.0)
    p_value = min(2.0 * min(p_low, p_high), 1.0)

    return {
        "value_a": float(value_a),
        "value_b": float(value_b),
        "observed_diff": float(observed_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }


def print_comparison(title, result, p=0.01):
    stars = "*" if result['p_value'] < p else " "
    print(
        f"  {title:<44} "
        f"A={result['value_a']:>9.3f}  B={result['value_b']:>9.3f}  "
        f"diff={result['observed_diff']:>+9.3f}  "
        f"95% CI=[{result['ci_low']:>+8.3f}, {result['ci_high']:>+8.3f}]  "
        f"p={result['p_value']:.4f} {stars}"
    )


def compare_label_metrics(dir_a, dir_b, n_resamples, ci, rng):
    print()
    print("Bifurcation count and total length by label, mean and variance")
    print("-" * 100)

    for label in LABELS:
        neurons_a = load_dataset(dir_a, label)
        neurons_b = load_dataset(dir_b, label)

        bif_a = [label_bifurcation_count(n) for n in neurons_a]
        bif_b = [label_bifurcation_count(n) for n in neurons_b]

        result = bs.bootstrap_pvalue_mean_diff(bif_a, bif_b, B=n_resamples)
        print_comparison(f"{label}: bifurcation count mean", result)

        result = bs.bootstrap_pvalue_var_ratio(bif_a, bif_b, B=n_resamples)
        print_comparison(f"{label}: bifurcation count variance", result)

        len_a = [label_total_length(n) for n in neurons_a]
        len_b = [label_total_length(n) for n in neurons_b]

        result = bs.bootstrap_pvalue_mean_diff(len_a, len_b, B=n_resamples)
        print_comparison(f"{label}: total length mean", result)

        result = bs.bootstrap_pvalue_var_ratio(len_a, len_b, B=n_resamples)
        print_comparison(f"{label}: total length variance", result)


def sholl_plot(roots, bin_size):
    """
    Sholl-style crossing counts restricted to sections with the given
    label: bin i counts how many label-matching sections pass through
    the interval [i * bin_size, (i + 1) * bin_size) at least once, using
    each section's full min-to-max distance range along its own path
    (not just individual segment direction), and counting a section at
    most once per bin even if its path revisits that bin.

    Distances are measured from the origin, not from a soma point: the
    soma is deleted by this pipeline (it's included in every label's
    LABEL_DELETE_SETS entry), and process_morphology's
    translate_sections already recenters every one of a neuron's roots
    at its own first point -- exactly the origin -- so every root
    already starts there after processing.

    Returns
    -------
    numpy.ndarray
        Length is one more than the furthest bin any label-matching
        section reaches; a neuron with no sections of this label
        returns an all-zero array of length 1.
    """
    tmp_sholl_plots = [r.sholl_plot(bin_size) for r in roots]
    max_len = max(sp.size for sp in tmp_sholl_plots)
    sholl_plots = []
    for sp in tmp_sholl_plots:
        tmp = np.zeros(max_len)
        tmp[:sp.size] = sp
        sholl_plots.append(tmp)
    return np.sum(sholl_plots, axis=0)


def compare_sholl_plots(dir_a, dir_b, bin_size, n_resamples, ci, rng):
    print()
    print(f"Sholl-plot crossings by bin, mean and variance, by label (bin_size={bin_size})")
    print("-" * 100)

    for label in LABELS:
        neurons_a = load_dataset(dir_a, label)
        neurons_b = load_dataset(dir_b, label)

        raw_a = [sholl_plot(n, bin_size) for n in neurons_a]
        raw_b = [sholl_plot(n, bin_size) for n in neurons_b]

        n_bins = max(len(arr) for arr in raw_a + raw_b)

        sholl_a = np.array([np.pad(arr, (0, n_bins - len(arr))) for arr in raw_a])
        sholl_b = np.array([np.pad(arr, (0, n_bins - len(arr))) for arr in raw_b])

        print()
        print(f"  {label}")

        for i in range(n_bins):
            bin_label = f"bin {i} [{i * bin_size:.0f}, {(i + 1) * bin_size:.0f})"

            result = bs.bootstrap_pvalue_mean_diff(sholl_a[:, i], sholl_b[:, i], B=n_resamples)
            print_comparison(f"{bin_label} mean", result, p=0.01 / n_bins)

            result = bs.bootstrap_pvalue_var_ratio(sholl_a[:, i], sholl_b[:, i], B=n_resamples)
            print_comparison(f"{bin_label} variance", result, p=0.01 / n_bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", help="Directory containing dataset A's .swc files.")
    parser.add_argument("dir_b", help="Directory containing dataset B's .swc files.")
    parser.add_argument("--bin-size", type=float, default=50.0, help="Sholl bin width (default: 10.0).")
    parser.add_argument("--n-resamples", type=int, default=10000, help="Number of bootstrap resamples (default: 10000).")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level for intervals (default: 0.95).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible bootstrap results.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Dataset A: {args.dir_a}")
    print(f"Dataset B: {args.dir_b}")
    print("(each label loaded and processed separately, matching extract_apc.py's own per-label delete_labels)")

    compare_label_metrics(args.dir_a, args.dir_b, args.n_resamples, args.ci, rng)
    compare_sholl_plots(args.dir_a, args.dir_b, args.bin_size, args.n_resamples, args.ci, rng)

    print()
    print("* marks a difference with bootstrap p < 0.05.")


if __name__ == "__main__":
    main()
