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
basal_dendrite separately (and apical_oblique too, unless
--merge-oblique-into-apical is given -- see below):

1. Bifurcation count and total length, comparing both the mean and the
   variance, one value per neuron.
2. Sholl-plot crossing counts, bin by bin, comparing both the mean and
   the variance across every neuron in each dataset.

Each comparison reports, per metric: each dataset's value, the observed
difference (A - B), a bootstrap confidence interval for that
difference, and a bootstrap p-value for whether it differs from zero.

Each comparison is seeded independently and deterministically (from
--seed plus that comparison's own label/metric/bin identity), rather
than sharing one rng threaded through the whole run. The variance
test's reject-and-redraw step consumes a variable, data-dependent
number of random draws, so with one shared rng, the order dir_a/dir_b
are given in -- or any upstream change in how many redraws an earlier,
unrelated comparison happened to need -- could shift every later
comparison onto a different random sequence, even though its own data
never changed. Independent seeding means a given comparison's p-value
depends only on its own data and --seed, never on any other
comparison's history in the same run.

--merge-oblique-into-apical stops apical_oblique from being deleted
out of the apical_dendrite tree (so its sections stay in and count
toward apical_dendrite's own bifurcation count, total length, and
Sholl crossings), and stops apical_oblique from being analyzed as its
own, separate label.

--min-nonzero (default 3) skips a Sholl bin's mean/variance test when
fewer than this many neurons, in EITHER group, have a nonzero crossing
count there. Near the far edge of a Sholl profile most neurons don't
reach that far and are zero-padded there, so with too few actual
(nonzero) values a bin's result is dominated by whichever one or two
neurons happen to reach furthest rather than reflecting the group as a
whole -- most visibly in the variance test, which such an outlier can
inflate by an order of magnitude while still returning a small,
"significant" bootstrap p-value.

--min-variance-ratio and --min-relative-mean-diff address a related
but different problem: with a large enough sample, even a practically
negligible difference can be statistically detectable (tiny p) purely
from sample size, not because the difference itself is meaningful.
Both default to None (no floor, i.e. the star follows the p-value
alone, as before). Set --min-variance-ratio (e.g. 1.5) to require the
larger variance be at least that many times the smaller one, and/or
--min-relative-mean-diff (e.g. 0.1) to require the mean difference be
at least that fraction of the larger group's value, before a result
with p < threshold is starred. A result that clears p but not the
effect-size floor is marked "s" (statistically detectable, small
effect) instead of "*", rather than looking identical to one that
cleared both or to one that wasn't significant at all.

Usage
-----
    python compare_datasets.py DIR_A DIR_B
    python compare_datasets.py DIR_A DIR_B --bin-size 20 --n-resamples 20000
    python compare_datasets.py DIR_A DIR_B --merge-oblique-into-apical
"""

import argparse
import hashlib

import numpy as np

from neuwalk.analysis.morphologies import load_morphologies
import bootstrap_test as bs


def make_rng(base_seed, *identity_parts):
    """
    A fresh, independently-seeded rng for one specific comparison,
    deterministic in both the run's base_seed (--seed) and a stable
    identity for this particular comparison (e.g. the label, metric,
    and bin). This is used instead of threading one shared rng through
    every comparison in the run: the variance test's reject-and-redraw
    loop consumes a variable, data-dependent number of random draws
    (however many attempts it takes to avoid a degenerate resample),
    and that count differs depending on which group ends up as x vs y
    -- so with one shared rng, swapping dir_a/dir_b (or any upstream
    change in how many redraws an earlier, unrelated comparison
    needed) shifts the shared rng's internal state, and every
    comparison after that point silently draws from a different random
    sequence even though its own data never changed. Seeding each
    comparison independently from its own identity means its result
    depends only on its own data and the run's base_seed -- never on
    the history of any other comparison in the same run.

    hashlib (not Python's built-in hash()) is used deliberately: the
    builtin is randomized per-process by default (PYTHONHASHSEED),
    which would make this non-reproducible across separate runs even
    with the same base_seed and the same identity_parts.
    """
    identity = "|".join(str(part) for part in identity_parts)
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    identity_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng([base_seed, identity_seed])

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


def load_dataset(directory, label, label_delete_sets=LABEL_DELETE_SETS):
    """
    Load every .swc file in directory for the given label, processed the
    same way extract_apc.py processes it (via load_morphologies with
    that label's own delete_labels). Returns a list of neurons, each
    itself a list of root sections -- there is no single soma root
    here, since "soma" is deleted for every label (see
    LABEL_DELETE_SETS), leaving each primary section of the target
    label as its own, separate root.
    """
    return load_morphologies(directory, delete_labels=label_delete_sets[label], soma_processing=False)



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





def print_comparison(title, result, p=0.01, min_effect=None):
    """
    min_effect, if given, is the minimum |observed_diff| a result must
    also reach (on top of p < p) to be starred -- a large enough
    sample can make even a practically negligible difference
    statistically detectable (tiny p), so the star is reserved for
    differences that are both statistically detectable AND at least
    this large. A result that clears p but not min_effect gets a
    distinct "s" marker (statistically detectable, small effect)
    instead of silently looking identical to one that cleared both,
    or to one that wasn't significant at all.
    """
    if result['p_value'] is None:
        # zero variance in at least one group -- bootstrap_pvalue_var_ratio
        # already printed why; report the observed values plainly, with
        # no CI/p-value to show since none was computed.
        print(
            f"  {title:<44} "
            f"A={result['value_a']:>9.3f}  B={result['value_b']:>9.3f}  "
            f"diff={result['observed_diff']:>+9.3f}  "
            f"95% CI=[undefined -- zero variance]  p=undefined"
        )
        return

    significant = result['p_value'] < p
    large_enough = min_effect is None or abs(result['observed_diff']) >= min_effect

    if significant and large_enough:
        marker = "*"
    elif significant:
        marker = "s"
    else:
        marker = " "

    print(
        f"  {title:<44} "
        f"A={result['value_a']:>9.3f}  B={result['value_b']:>9.3f}  "
        f"diff={result['observed_diff']:>+9.3f}  "
        f"95% CI=[{result['ci_low']:>+8.3f}, {result['ci_high']:>+8.3f}]  "
        f"p={result['p_value']:.4f} {marker}"
    )


def mean_diff_min_effect(result, min_relative_mean_diff):
    """
    The mean-diff test's min_effect, scaled to that comparison's own
    values: min_relative_mean_diff (a fraction) times the larger of
    the two observed values, so e.g. --min-relative-mean-diff 0.1
    requires at least a 10% difference relative to the larger group,
    not a fixed absolute number that would mean something different
    for a bifurcation count than for a total length in microns.
    """
    if min_relative_mean_diff is None:
        return None
    return min_relative_mean_diff * max(abs(result['value_a']), abs(result['value_b']))


def compare_label_metrics(dir_a, dir_b, n_resamples, ci, base_seed, labels=LABELS, label_delete_sets=LABEL_DELETE_SETS,
                           min_variance_ratio=None, min_relative_mean_diff=None):
    print()
    print("Bifurcation count and total length by label, mean and variance")
    print("-" * 100)

    for label in labels:
        neurons_a = load_dataset(dir_a, label, label_delete_sets)
        neurons_b = load_dataset(dir_b, label, label_delete_sets)

        if len(neurons_a) <= 2 or len(neurons_b) <= 2:
            print(f"{label} cannot be analyzed due to lack of samples.")
            continue

        bif_a = [label_bifurcation_count(n) for n in neurons_a]
        bif_b = [label_bifurcation_count(n) for n in neurons_b]

        min_log_ratio = np.log(min_variance_ratio) if min_variance_ratio is not None else None

        result = bs.bootstrap_pvalue_mean_diff(make_rng(base_seed, label, "bifurcation_count", "mean"), bif_a, bif_b, B=n_resamples)
        print_comparison(f"{label}: bifurcation count mean", result, min_effect=mean_diff_min_effect(result, min_relative_mean_diff))

        result = bs.bootstrap_pvalue_var_ratio(make_rng(base_seed, label, "bifurcation_count", "variance"), bif_a, bif_b, B=n_resamples)
        print_comparison(f"{label}: bifurcation count variance", result, min_effect=min_log_ratio)

        len_a = [label_total_length(n) for n in neurons_a]
        len_b = [label_total_length(n) for n in neurons_b]

        result = bs.bootstrap_pvalue_mean_diff(make_rng(base_seed, label, "total_length", "mean"), len_a, len_b, B=n_resamples)
        print_comparison(f"{label}: total length mean", result, min_effect=mean_diff_min_effect(result, min_relative_mean_diff))

        result = bs.bootstrap_pvalue_var_ratio(make_rng(base_seed, label, "total_length", "variance"), len_a, len_b, B=n_resamples)
        print_comparison(f"{label}: total length variance", result, min_effect=min_log_ratio)


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


def compare_sholl_plots(dir_a, dir_b, bin_size, n_resamples, ci, base_seed, labels=LABELS, label_delete_sets=LABEL_DELETE_SETS, min_nonzero=3,
                         min_variance_ratio=None, min_relative_mean_diff=None):

        print()
        print(f"Sholl-plot crossings by bin, mean and variance, by label (bin_size={bin_size})")
        print("-" * 100)

        for label in labels:
            try:
                neurons_a = load_dataset(dir_a, label, label_delete_sets)
                neurons_b = load_dataset(dir_b, label, label_delete_sets)

                if not neurons_a or not neurons_b:
                    # one (or both) datasets have no neurons with this
                    # label at all -- e.g. a preset with no oblique
                    # dendrites, or a label that only exists in one of
                    # the two directories. There's nothing to compare,
                    # and left unguarded this produces an empty (1D
                    # rather than (0, n_bins)) array a few lines below,
                    # crashing on the first indexed slice.
                    print(f"\n  {label:<44} skipped -- no neurons with this label in at least one dataset "
                          f"(n: A={len(neurons_a)}, B={len(neurons_b)})")
                    continue

                raw_a = [sholl_plot(n, bin_size) for n in neurons_a]
                raw_b = [sholl_plot(n, bin_size) for n in neurons_b]

                n_bins = max(len(arr) for arr in raw_a + raw_b)

                sholl_a = np.array([np.pad(arr, (0, n_bins - len(arr))) for arr in raw_a])
                sholl_b = np.array([np.pad(arr, (0, n_bins - len(arr))) for arr in raw_b])

                print()
                print(f"  {label}")

                for i in range(n_bins):
                    bin_label = f"bin {i} [{i * bin_size:.0f}, {(i + 1) * bin_size:.0f})"

                    # Near the far edge of a Sholl profile, most neurons don't
                    # reach that far and are zero-padded there -- only the
                    # neurons that DO reach contribute real information at
                    # this bin. With too few of those, a bin's mean/variance
                    # is dominated by whichever one or two neurons happen to
                    # reach furthest, rather than reflecting the group as a
                    # whole -- skip the test there instead of reporting a
                    # technically-correct but practically-meaningless result.
                    nonzero_a = np.count_nonzero(sholl_a[:, i])
                    nonzero_b = np.count_nonzero(sholl_b[:, i])

                    if nonzero_a < min_nonzero or nonzero_b < min_nonzero:
                        print(f"  {bin_label:<44} skipped -- fewer than {min_nonzero} neurons reach this "
                              f"bin in at least one group (nonzero: A={nonzero_a}, B={nonzero_b})")
                        continue

                    min_log_ratio = np.log(min_variance_ratio) if min_variance_ratio is not None else None

                    result = bs.bootstrap_pvalue_mean_diff(make_rng(base_seed, label, "sholl", i, "mean"), sholl_a[:, i], sholl_b[:, i], B=n_resamples)
                    print_comparison(f"{bin_label} mean", result, p=0.01 / n_bins, min_effect=mean_diff_min_effect(result, min_relative_mean_diff))

                    result = bs.bootstrap_pvalue_var_ratio(make_rng(base_seed, label, "sholl", i, "variance"), sholl_a[:, i], sholl_b[:, i], B=n_resamples)
                    print_comparison(f"{bin_label} variance", result, p=0.01 / n_bins, min_effect=min_log_ratio)
            except Exception as error:
                # a catch-all safety net, not the primary guard (that's
                # the explicit neurons_a/neurons_b check above) -- this
                # is for any other, unforeseen failure specific to one
                # label, so it doesn't take the rest of the comparison
                # down with it.
                print(f"\n  {label:<44} skipped -- failed to compare for Sholl plots ({type(error).__name__}: {error})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", help="Directory containing dataset A's .swc files.")
    parser.add_argument("dir_b", help="Directory containing dataset B's .swc files.")
    parser.add_argument("--bin-size", type=float, default=50.0, help="Sholl bin width (default: 50.0).")
    parser.add_argument("--n-resamples", type=int, default=10000, help="Number of bootstrap resamples (default: 10000).")
    parser.add_argument("--ci", type=float, default=0.99, help="Confidence level for intervals (default: 0.95).")
    parser.add_argument("--seed", type=int, default=56, help="Random seed, for reproducible bootstrap results.")
    parser.add_argument("--merge-oblique-into-apical", action="store_true",
                         help="Don't delete apical_oblique when loading apical_dendrite (so its sections stay in "
                              "the apical tree and count toward apical_dendrite's own statistics), and don't "
                              "analyze apical_oblique as its own, separate label.")
    parser.add_argument("--min-nonzero", type=int, default=3,
                         help="Minimum number of neurons that must have a nonzero Sholl crossing count at a given "
                              "bin, in EACH group, for that bin's mean/variance test to be performed (default: 3). "
                              "Below this, a bin's result is dominated by whichever one or two neurons happen to "
                              "reach that far rather than reflecting the group as a whole, so it's skipped instead.")
    parser.add_argument("--min-variance-ratio", type=float, default=None,
                         help="Minimum ratio between the two groups' variances (the larger over the smaller) for a "
                              "variance-test result to be starred as significant, on top of clearing the p-value "
                              "threshold (default: None, i.e. no floor). With a large enough sample, even a "
                              "practically negligible variance difference can be statistically detectable; a result "
                              "with p < threshold but below this ratio is marked 's' (statistically detectable, "
                              "small effect) instead of '*'.")
    parser.add_argument("--min-relative-mean-diff", type=float, default=None,
                         help="Minimum difference between the two groups' means, as a fraction of the larger "
                              "group's value, for a mean-test result to be starred as significant, on top of "
                              "clearing the p-value threshold (default: None, i.e. no floor). Same reasoning as "
                              "--min-variance-ratio, applied to the mean-difference test instead.")
    args = parser.parse_args()

    if args.merge_oblique_into_apical:
        labels = tuple(label for label in LABELS if label != "apical_oblique")
        label_delete_sets = dict(LABEL_DELETE_SETS)
        label_delete_sets["apical_dendrite"] = [
            label for label in LABEL_DELETE_SETS["apical_dendrite"] if label != "apical_oblique"
        ]
    else:
        labels = LABELS
        label_delete_sets = LABEL_DELETE_SETS

    print(f"Dataset A: {args.dir_a}")
    print(f"Dataset B: {args.dir_b}")

    compare_label_metrics(args.dir_a, args.dir_b, args.n_resamples, args.ci, args.seed, labels, label_delete_sets,
                           args.min_variance_ratio, args.min_relative_mean_diff)
    compare_sholl_plots(args.dir_a, args.dir_b, args.bin_size, args.n_resamples, args.ci, args.seed, labels, label_delete_sets, args.min_nonzero,
                         args.min_variance_ratio, args.min_relative_mean_diff)

    print()
    print("* marks a difference with bootstrap p < 0.01.")
    if args.min_variance_ratio is not None or args.min_relative_mean_diff is not None:
        print("s marks a difference with bootstrap p < 0.01 but below the minimum effect size -- statistically "
              "detectable, but too small to flag as a meaningful difference.")


if __name__ == "__main__":
    main()
