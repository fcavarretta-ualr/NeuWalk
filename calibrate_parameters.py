"""
Iteratively calibrate a preset's *.parameters.json against its own
experimental targets.

For each label (apical_dendrite, basal_dendrite), this script creates
ONE persistent TopologySynthesizer per seed -- not a fresh instance per
bin or per iteration. Each of these trees is grown bin by bin, left to
right (closest to the soma first), using synthesize(distance_limit=...)
to advance incrementally from wherever it currently is, exactly the way
TopologySynthesizer.synthesize_progressive grows a tree internally.

Growing a bin validates each tree INDIVIDUALLY, exactly the way
synthesize_progressive itself does (_sholl_status/_regenerate_window):
if a tree's own Sholl count at that bin falls outside
working_mean +/- n_std*working_std, that ONE tree is rolled back
(undo_synthesize) and regrown, up to --max-attempts-per-window times,
before being accepted regardless. This matters because
synthesize_topologies/synthesize_progressive (what a preset's real
generation, and scripts like test_3.py, actually use) applies this
same per-tree rejection-and-retry -- without it here too, this script
would calibrate parameters against a different, uncoditioned
distribution than what real generation actually produces, and the
resulting statistics would disagree with test_3.py's even though both
started from the same calibrated JSON.

After every tree has been grown (and individually validated) through a
given bin, the simulated Sholl mean at that bin (averaged across all
seeds) is compared against the TRUE, ORIGINAL experimental mean (from
the input JSON, never changed). If the difference exceeds
--sholl-threshold:

  1. every tree's growth for that bin is undone via undo_synthesize()
     -- not discarded and rebuilt, just rolled back one bin;
  2. the WORKING Sholl mean for that bin is shifted by the observed
     error;
  3. rates are refit ONCE from the shifted mean (event_rates is
     deterministic given the same input statistics, so there's no need
     to solve it separately per seed), and every tree's
     main_event_sampler/event_sampler is replaced in place with a
     fresh EventSampler built from those shared rates (each keeping its
     own rng, so seeds stay independent) -- no TopologySynthesizer is
     ever discarded or recreated;
  4. that bin is regrown (with the same per-tree validation) for every
     tree and retested, until it converges or --max-iterations is
     reached.

Once every bin for a label is calibrated this way, the same shift-and-
retest loop runs once more for that label's bifurcation_count mean,
with the same per-tree validation principle applied at the whole-tree
level: after growing a tree to its full extent, if its own
bifurcation_count() falls outside working_mean +/- n_std*working_std,
that tree alone is rolled all the way back (repeated undo_synthesize()
until its synthesis_logs is empty) and regrown from scratch, up to
--max-attempts-per-window times, matching
synthesize_progressive's own _bifurcation_count_status logic. Since
bifurcation_count constrains rates jointly across every bin (not just
one), a THRESHOLD crossing at the group level still rolls every tree
back to its very start (not just one bin) and regrows the whole tree
in a single synthesize(distance_limit=...) call per tree once rates
are refit.

The result -- a working copy of the input JSON with adjusted
sholl_plot means and bifurcation_count means -- is written to
--output. Nothing else in the JSON (std values, primary_count_range,
bin_size, no_bifurcation_bins, etc.) is touched.

Usage
-----
    python calibrate_parameters.py neuwalk/presets/anterior_piriform_cortex/semilunar.parameters.json
    python calibrate_parameters.py PARAMS.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 --n-seeds 200
"""

import argparse
import copy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from neuwalk.random import Random
from neuwalk.synthesis import TopologySynthesizer
from neuwalk.synthesis.topology.sampling import EventSampler
from neuwalk.synthesis.topology.sampling.estimation import event_rates, initial_count_pmf


LABELS = ("apical_dendrite", "basal_dendrite", "apical_oblique")

# A shifted mean is clamped to this instead of exactly 0, since a Sholl
# mean of exactly 0 truncates the fitting process at that bin entirely
# (event_rates stops at the first zero mean), which would silently cut
# off every bin after it.
MIN_MEAN = 0.1


def load_params(path):
    return json.loads(Path(path).read_text())


def make_trees(label_params, label, n_seeds, step_size):
    """
    Create one persistent TopologySynthesizer per seed.

    Only the FIRST tree is constructed the normal way (with sholl_plot=
    passed through, letting EventSampler solve event_rates for it).
    Every other tree reuses that first solve's bifurcation_density/
    annihilation_density directly, instead of each independently
    triggering its own redundant event_rates call.

    This matters because event_rates is deterministic given the same
    label_params -- it doesn't depend on the seed at all -- so solving
    it once per seed just to construct n_seeds identical-in-expectation
    trees is pure waste. For a label with few Sholl bins the solve is
    cheap and this waste is negligible, but for one with many bins
    (e.g. neocortex's apical_dendrite, ~106 bins vs. anterior_piriform_
    cortex's ~9), the nonlinear (Ipopt) solve itself is far more
    expensive, and redundantly re-solving it n_seeds times before any
    bin-by-bin calibration even begins can dominate total runtime on
    its own.

    A tree built this way can't also be given sholl_plot= or
    primary_count_range= directly (TopologySynthesizer forwards both
    unconditionally to EventSampler, which rejects sholl_plot together
    with direct densities) -- so primary_count_min/init_count_cdf,
    which primary_count_range would normally set up, are copied
    directly from the first tree's EventSampler instead. This is safe
    because they're deterministic given the same sholl_plot/
    primary_count_range (no randomness in computing them, only in
    later sampling from them) and are plain attributes, not properties
    with validation.

    label_params is filtered down to TopologySynthesizer's actual
    parameters before being unpacked: some presets' JSON carries extra
    keys used elsewhere in the full generation pipeline but not
    accepted by TopologySynthesizer.__init__ at all -- e.g.
    neocortex's apical_dendrite has a bifurcation_internal_density
    entry (used by connect_internal_branches for oblique grafting in
    _generation.py), which isn't one of TopologySynthesizer's
    parameters and would otherwise raise a TypeError here.
    """
    valid_keys = {
        "sholl_plot", "bifurcation_count", "primary_count_range",
        "no_bifurcation_bins", "no_annihilation_bins", "bin_size",
    }
    label_params = {key: value for key, value in label_params.items() if key in valid_keys}

    other_kwargs = {
        key: value
        for key, value in label_params.items()
        if key not in ("sholl_plot", "primary_count_range")
    }

    first_tree = TopologySynthesizer(
        Random(0),
        step_size=step_size,
        label=label,
        with_soma=(label != "apical_oblique"),
        **label_params,
    )

    trees = [first_tree]
    template_sampler = first_tree.main_event_sampler

    for seed in range(1, n_seeds):
        tree = TopologySynthesizer(
            Random(seed),
            step_size=step_size,
            label=label,
            with_soma=(label != "apical_oblique"),
            bifurcation_density=template_sampler.bifurcation_density,
            annihilation_density=template_sampler.annihilation_density,
            **other_kwargs,
        )
        tree.main_event_sampler.primary_count_min = template_sampler.primary_count_min
        tree.main_event_sampler.init_count_cdf = template_sampler.init_count_cdf
        trees.append(tree)

    return trees


def refit_event_samplers(trees, label_params, step_size, bin_size):
    """
    Refit event rates ONCE from label_params' current sholl_plot mean
    (and bifurcation_count, if set), then give every tree in trees its
    own fresh EventSampler built from those shared rates -- each with
    its own tree's rng, so seeds stay independent -- replacing
    main_event_sampler/event_sampler in place. No TopologySynthesizer
    is created or discarded here.

    If primary_count_range is set, primary_count_min/init_count_cdf are
    also recomputed ONCE here and copied onto every tree's new sampler.
    This matters even though a tree that has already initialized its
    primary sections doesn't need them again: calibrate_bifurcation_count
    rolls a tree all the way back to before it ever initialized, and
    the next synthesize() call re-triggers initialize(), which calls
    sample_primary_section_count() again -- if the replacement sampler
    built here lacked these (as it would if only bifurcation_density/
    annihilation_density were passed to EventSampler, without also
    setting these up), that call would fail with "The primary-count
    distribution is unavailable." the next time a rolled-back tree
    tries to reinitialize.
    """
    rates = event_rates(
        bin_size,
        label_params["sholl_plot"],
        step_size,
        bifurcation_count=label_params.get("bifurcation_count"),
        no_bifurcation_bins=label_params.get("no_bifurcation_bins"),
        no_annihilation_bins=label_params.get("no_annihilation_bins"),
    )

    primary_count_range = label_params.get("primary_count_range")
    primary_count_min = None
    init_count_cdf = None

    if primary_count_range is not None:
        primary_count_min = int(primary_count_range["min"])
        sholl_plot = label_params["sholl_plot"]
        probabilities = initial_count_pmf(
            sholl_plot["mean"][0], sholl_plot["std"][0],
            primary_count_range["min"], primary_count_range["max"],
        )
        init_count_cdf = np.cumsum(probabilities)

    for tree in trees:
        sampler = EventSampler(
            rng=tree.rng,
            step_size=step_size,
            bin_size=bin_size,
            bifurcation_density=rates["bifurcation_rate"],
            annihilation_density=rates["annihilation_rate"],
        )
        sampler.primary_count_min = primary_count_min
        sampler.init_count_cdf = init_count_cdf
        tree.main_event_sampler = sampler
        tree.event_sampler = sampler


def _is_within_tolerance(generated, target_mean, target_std, n_std):
    """mean +/- n_std*std, matching synthesize_progressive's _sholl_status/_bifurcation_count_status exactly."""
    if np.isclose(target_std, 0.0):
        return np.isclose(generated, target_mean)
    return (target_mean - n_std * target_std) <= generated <= (target_mean + n_std * target_std)


def _grow_one_tree_with_retry(tree, distance_limit, bin_index, target_mean, target_std, n_std, max_attempts_per_window, bin_size):
    """
    Grow one tree to distance_limit (covering bin_index). If this
    tree's OWN Sholl count at bin_index falls outside
    target_mean +/- n_std*target_std, roll back just this bin
    (undo_synthesize) and retry, up to max_attempts_per_window times --
    matching synthesize_progressive's own per-bin, per-tree validation
    (_sholl_status/_regenerate_window) exactly, just applied
    independently to each persistent tree here rather than to a single
    tree inside one synthesize_progressive call.

    If every attempt fails, the last attempt's growth is kept rather
    than rolled back, so the tree is never left without this bin grown
    at all. Unlike synthesize_progressive, this does not expand the
    rollback window to re-attempt earlier bins too -- the outer
    calibration loop's own mean-shifting is relied on instead to make
    a persistently-failing bin easier to satisfy for the whole group.

    Returns True if some attempt passed validation, False if every
    attempt was exhausted without passing (the tree is still grown
    either way).
    """
    for attempt in range(max_attempts_per_window):
        tree.synthesize(distance_limit=distance_limit)
        generated = tree.sholl_plot(max_distance=bin_index * bin_size)[bin_index]

        if _is_within_tolerance(generated, target_mean, target_std, n_std):
            return True

        if attempt < max_attempts_per_window - 1:
            tree.undo_synthesize()

    return False


def grow_trees(trees, distance_limit, bin_index, target_mean, target_std, n_std, max_attempts_per_window, bin_size, max_workers):
    """
    Advance every tree to distance_limit (from wherever each currently
    is), in parallel -- each tree independently retried (rolled back
    and regrown, up to max_attempts_per_window times) if its own Sholl
    count at bin_index falls outside target_mean +/- n_std*target_std,
    matching synthesize_progressive's own per-tree validation.

    Returns the number of trees that never passed validation within
    max_attempts_per_window (kept anyway, with their last attempt's
    growth), for diagnostic reporting.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _grow_one_tree_with_retry, tree, distance_limit, bin_index,
                target_mean, target_std, n_std, max_attempts_per_window, bin_size,
            )
            for tree in trees
        ]

        passed = [future.result() for future in futures]  # re-raises any exception too

    return sum(1 for ok in passed if not ok)


def _grow_one_tree_full_with_retry(tree, full_distance_limit, target_mean, target_std, n_std, max_attempts_per_window):
    """
    Grow one tree to its full extent (full_distance_limit). If this
    tree's OWN bifurcation_count() falls outside
    target_mean +/- n_std*target_std, roll it all the way back (repeated
    undo_synthesize() until synthesis_logs is empty) and regrow from
    scratch, up to max_attempts_per_window times -- matching
    synthesize_progressive's own _bifurcation_count_status/whole-tree
    reset logic, just applied independently per persistent tree here.

    Returns True if some attempt passed validation, False if every
    attempt was exhausted without passing (the tree is still grown to
    its full extent either way).
    """
    for attempt in range(max_attempts_per_window):
        tree.synthesize(distance_limit=full_distance_limit)
        generated = tree.bifurcation_count()

        if _is_within_tolerance(generated, target_mean, target_std, n_std):
            return True

        if attempt < max_attempts_per_window - 1:
            while len(tree.synthesis_logs) > 0:
                tree.undo_synthesize()

    return False


def grow_trees_full(trees, full_distance_limit, target_mean, target_std, n_std, max_attempts_per_window, max_workers):
    """
    Grow every tree to its full extent, in parallel -- each tree
    independently retried (fully rolled back and regrown, up to
    max_attempts_per_window times) if its own bifurcation_count() falls
    outside target_mean +/- n_std*target_std.

    Returns the number of trees that never passed validation within
    max_attempts_per_window (kept anyway), for diagnostic reporting.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _grow_one_tree_full_with_retry, tree, full_distance_limit,
                target_mean, target_std, n_std, max_attempts_per_window,
            )
            for tree in trees
        ]

        passed = [future.result() for future in futures]

    return sum(1 for ok in passed if not ok)


def rollback_trees(trees, target_log_count=None):
    """
    Undo the most recent synthesize() call for every tree (one bin's
    worth of growth), or -- if target_log_count is given -- undo
    repeatedly until each tree's synthesis_logs shrinks to that length
    (0 for a full reset back to before any growth at all).
    """
    for tree in trees:
        if target_log_count is None:
            tree.undo_synthesize()
        else:
            while len(tree.synthesis_logs) > target_log_count:
                tree.undo_synthesize()


def simulated_sholl_mean_at_bin(trees, bin_index):
    values = [
        tree.sholl_plot(max_distance=bin_index * tree.bin_size)[bin_index]
        for tree in trees
    ]
    return float(np.mean(values))


def simulated_bifurcation_mean(trees):
    return float(np.mean([tree.bifurcation_count() for tree in trees]))


def calibrate_sholl_bin(trees, label_params, bin_index, bin_size, target_mean, threshold,
                         max_iterations, step_size, n_std, max_attempts_per_window, max_workers):
    """
    Shift label_params['sholl_plot']['mean'][bin_index] until the
    simulated mean at that bin (averaged across all persistent trees)
    is within threshold of target_mean, or max_iterations is reached.
    Mutates label_params in place. Every tree is grown/rolled back
    together, as one group -- no tree is ever discarded or recreated.

    Each tree's own growth at this bin is separately validated against
    the CURRENT WORKING mean/std (not the original target): this is
    what synthesize_progressive itself would validate against if run
    with label_params as they stand right now, so per-tree validation
    here needs to match that, not the final target this bin is being
    calibrated towards.
    """
    distance_limit = bin_size * bin_index

    for iteration in range(1, max_iterations + 1):
        working_mean = label_params["sholl_plot"]["mean"][bin_index]
        working_std = label_params["sholl_plot"]["std"][bin_index]

        n_failed = grow_trees(trees, distance_limit, bin_index, working_mean, working_std, n_std, max_attempts_per_window, bin_size, max_workers)
        if n_failed:
            print(f"    bin {bin_index}, iteration {iteration}: {n_failed}/{len(trees)} tree(s) did not pass per-tree validation within {max_attempts_per_window} attempt(s) (kept anyway).")

        simulated = simulated_sholl_mean_at_bin(trees, bin_index)
        diff = target_mean - simulated
        current = working_mean

        print(f"    bin {bin_index}, iteration {iteration}: working_mean={current:.3f} simulated={simulated:.3f} target={target_mean:.3f} diff={diff:+.3f}")

        if abs(diff) <= threshold:
            print(f"    bin {bin_index}: converged after {iteration} iteration(s).")
            return

        # roll back just this bin's growth for every tree, then shift
        # the working mean and refit rates in place before retrying
        rollback_trees(trees)
        label_params["sholl_plot"]["mean"][bin_index] = max(MIN_MEAN, current + diff)
        refit_event_samplers(trees, label_params, step_size, bin_size)
    else:
        print(f"    bin {bin_index}: did NOT converge within {max_iterations} iterations; keeping the last value.")


def calibrate_bifurcation_count(trees, label_params, bin_size, full_distance_limit, target_mean, threshold,
                                 max_iterations, step_size, n_std, max_attempts_per_window, max_workers):
    """
    Shift label_params['bifurcation_count']['mean'] until the simulated
    mean bifurcation count (averaged across all persistent trees) is
    within threshold of target_mean, or max_iterations is reached.

    Each tree's own bifurcation_count() is separately validated against
    the CURRENT WORKING mean/std (matching synthesize_progressive's own
    _bifurcation_count_status): a tree outside tolerance is rolled all
    the way back and regrown from scratch, up to max_attempts_per_window
    times, on every iteration -- including the first, even though the
    trees are already grown to their full extent by calibrate_sholl_bin
    at that point (that growth was only validated bin by bin against
    Sholl targets, never against bifurcation_count).

    Unlike calibrate_sholl_bin, a group-level threshold crossing here
    rolls every tree back to its very start (not just one bin):
    bifurcation_count constrains rates jointly across the whole tree,
    so once it shifts, every bin's rates -- and therefore the whole
    tree -- need regrowing, not just the most recent bin.
    """
    for iteration in range(1, max_iterations + 1):
        working_mean = label_params["bifurcation_count"]["mean"]
        working_std = label_params["bifurcation_count"]["std"]

        n_failed = grow_trees_full(trees, full_distance_limit, working_mean, working_std, n_std, max_attempts_per_window, max_workers)
        if n_failed:
            print(f"    iteration {iteration}: {n_failed}/{len(trees)} tree(s) did not pass per-tree validation within {max_attempts_per_window} attempt(s) (kept anyway).")

        simulated = simulated_bifurcation_mean(trees)
        diff = target_mean - simulated
        current = working_mean

        print(f"    iteration {iteration}: working_mean={current:.3f} simulated={simulated:.3f} target={target_mean:.3f} diff={diff:+.3f}")

        if abs(diff) <= threshold:
            print(f"    converged after {iteration} iteration(s).")
            return

        rollback_trees(trees, target_log_count=0)
        label_params["bifurcation_count"]["mean"] = max(MIN_MEAN, current + diff)
        refit_event_samplers(trees, label_params, step_size, bin_size)
    else:
        print(f"    did NOT converge within {max_iterations} iterations; keeping the last value.")


def calibrate(original_params, sholl_threshold, bifurcation_threshold, n_seeds, max_iterations,
              step_size, n_std, max_attempts_per_window, max_workers):
    working_params = copy.deepcopy(original_params)

    for label in LABELS:
        if label not in working_params:
            continue

        bin_size = original_params[label]["bin_size"]
        n_bins = len(original_params[label]["sholl_plot"]["mean"])

        trees = make_trees(working_params[label], label, n_seeds, step_size)

        print(f"\n=== {label}: Sholl plot, bin by bin ===")
        for bin_index in range(n_bins):
            target_mean = original_params[label]["sholl_plot"]["mean"][bin_index]
            calibrate_sholl_bin(
                trees, working_params[label], bin_index, bin_size, target_mean, sholl_threshold,
                max_iterations, step_size, n_std, max_attempts_per_window, max_workers,
            )

        print(f"\n=== {label}: bifurcation count ===")
        target_mean = original_params[label]["bifurcation_count"]["mean"]
        full_distance_limit = bin_size * (n_bins - 1)
        calibrate_bifurcation_count(
            trees, working_params[label], bin_size, full_distance_limit, target_mean, bifurcation_threshold,
            max_iterations, step_size, n_std, max_attempts_per_window, max_workers,
        )

    return working_params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parameters_json", help="Path to a preset's *.parameters.json file.")
    parser.add_argument("--output", default=None, help="Path to write the corrected JSON (default: <input>.corrected.json).")
    parser.add_argument("--sholl-threshold", type=float, required=True, help="Max allowed |experimental - simulated| for each Sholl bin before shifting.")
    parser.add_argument("--bifurcation-threshold", type=float, required=True, help="Max allowed |experimental - simulated| for bifurcation count before shifting.")
    parser.add_argument("--n-seeds", type=int, default=200, help="Number of persistent trees (one per seed) synthesized per label (default: 200).")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max shift-and-retest iterations per bin/metric (default: 20).")
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--n-std", type=float, default=3.0, help="Per-tree tolerance (mean +/- n_std*std) for accepting a tree's own Sholl/bifurcation_count value, matching synthesize_progressive's own validation (default: 3.0).")
    parser.add_argument("--max-attempts-per-window", type=int, default=10, help="Per-tree retry attempts before accepting its last attempt regardless, matching synthesize_progressive's own retry limit (default: 10).")
    parser.add_argument("--max-workers", type=int, default=None, help="Thread pool size per growth step (default: os.cpu_count()).")
    args = parser.parse_args()

    original_params = load_params(args.parameters_json)

    corrected = calibrate(
        original_params, args.sholl_threshold, args.bifurcation_threshold, args.n_seeds, args.max_iterations,
        args.step_size, args.n_std, args.max_attempts_per_window, args.max_workers,
    )

    output_path = args.output or str(Path(args.parameters_json).with_suffix(".corrected.json"))
    Path(output_path).write_text(json.dumps(corrected, indent=2))
    print(f"\nwrote corrected parameters to {output_path}")


if __name__ == "__main__":
    main()
