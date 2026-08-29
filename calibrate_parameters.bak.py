"""
Calibrate a preset's *.parameters.json against its own experimental
targets, using TopologySynthesizer.synthesize_progressive directly
(with its distance_limit parameter) -- not a local copy of it.

For each label present in the JSON (apical_dendrite, basal_dendrite,
apical_oblique), in order:

1. Sholl plot, bin by bin, left to right. For each bin: build
   --n-generations fresh TopologySynthesizer instances
   (Random(0)..Random(n_generations-1)) and grow every one, in
   parallel across a thread pool, via
   synthesize_progressive(distance_limit=bin_size*bin_index) -- so
   only that bin and everything before it is generated and validated.
   Compare the average simulated Sholl count at that bin against the
   experimental mean for that bin (from the input JSON, never
   changed). If the difference is larger than --sholl-threshold, shift
   the working Sholl mean for that bin by the observed difference and
   rebuild and retest with brand new trees, repeating until it
   converges or --max-iterations is reached.

2. Bifurcation count, once every bin is calibrated. Build
   --n-generations fresh trees and grow each one fully
   (distance_limit=None), which naturally includes
   synthesize_progressive's own bifurcation-count check once a tree
   reaches full extent. Shift the working bifurcation_count mean the
   same way, rebuilding fresh trees each iteration, until it converges
   or --max-iterations is reached.

Only sholl_plot mean and bifurcation_count mean are ever changed.
Every std value, and everything else in the JSON (primary_count_range,
bin_size, no_bifurcation_bins, no_annihilation_bins, etc.), is left
untouched.

Usage
-----
    python calibrate_parameters.py neuwalk/presets/anterior_piriform_cortex/semilunar.parameters.json
    python calibrate_parameters.py PARAMS.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 --n-std 2.5
"""

import argparse
import copy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from neuwalk.random import Random
from neuwalk.synthesis import TopologySynthesizer


LABELS = ("apical_dendrite", "basal_dendrite", "apical_oblique")

# A shifted mean is clamped to this instead of exactly 0, since a Sholl
# or bifurcation-count mean of exactly 0 breaks event_rates' fitting
# (it stops at the first zero Sholl mean, silently truncating every
# bin after it).
MIN_MEAN = 0.1


def load_params(path):
    return json.loads(Path(path).read_text())


def _filter_to_topology_synthesizer_kwargs(label_params):
    """
    Keep only the keys TopologySynthesizer.__init__ actually accepts.
    Some presets' JSON carries extra keys used elsewhere in the full
    generation pipeline but not accepted here -- e.g. neocortex's
    apical_dendrite has a bifurcation_internal_density entry (used by
    connect_internal_branches for oblique grafting), which would
    otherwise raise a TypeError.
    """
    valid_keys = {
        "sholl_plot", "bifurcation_count", "primary_count_range",
        "no_bifurcation_bins", "no_annihilation_bins", "bin_size",
    }
    return {key: value for key, value in label_params.items() if key in valid_keys}


def _build_and_grow_one_tree(label_params, label, seed, step_size, n_std, max_attempts_per_window, max_total_attempts, distance_limit):
    """Construct one fresh TopologySynthesizer and grow it via its own synthesize_progressive. Runs in a worker thread."""
    filtered_params = _filter_to_topology_synthesizer_kwargs(label_params)

    tree = TopologySynthesizer(
        Random(seed),
        step_size=step_size,
        label=label,
        with_soma=(label != "apical_oblique"),
        **filtered_params,
    )

    tree.synthesize_progressive(
        n_std=n_std,
        max_attempts_per_window=max_attempts_per_window,
        max_total_attempts=max_total_attempts,
        distance_limit=distance_limit,
        verbose=False,
    )

    return tree


def build_and_grow_trees(label_params, label, n_generations, step_size, n_std, max_attempts_per_window, max_total_attempts,
                          distance_limit, max_workers):
    """Build and grow n_generations fresh trees in parallel; return the list of grown trees."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(
            lambda seed: _build_and_grow_one_tree(
                label_params, label, seed, step_size, n_std, max_attempts_per_window,
                max_total_attempts, distance_limit,
            ),
            range(n_generations),
        ))


def simulated_sholl_mean_at_bin(trees, bin_index, bin_size):
    values = [tree.sholl_plot(max_distance=bin_index * bin_size)[bin_index] for tree in trees]
    return float(np.mean(values))


def simulated_bifurcation_mean(trees):
    return float(np.mean([tree.bifurcation_count() for tree in trees]))


def calibrate_sholl_bin(label_params, label, bin_index, bin_size, target_mean, threshold, n_generations, max_iterations,
                         step_size, n_std, max_attempts_per_window, max_total_attempts, max_workers):
    """
    Shift label_params['sholl_plot']['mean'][bin_index] until the
    average simulated Sholl count at that bin, across n_generations
    FRESH trees each grown via synthesize_progressive up to
    distance_limit = bin_size * bin_index, is within threshold of
    target_mean, or max_iterations is reached. Mutates label_params in
    place; std is never touched.
    """
    distance_limit = bin_size * bin_index

    for iteration in range(1, max_iterations + 1):
        trees = build_and_grow_trees(
            label_params, label, n_generations, step_size, n_std, max_attempts_per_window,
            max_total_attempts, distance_limit, max_workers,
        )

        simulated = simulated_sholl_mean_at_bin(trees, bin_index, bin_size)
        diff = target_mean - simulated
        current = label_params["sholl_plot"]["mean"][bin_index]

        print(f"    bin {bin_index}, iteration {iteration}: working_mean={current:.3f} simulated={simulated:.3f} target={target_mean:.3f} diff={diff:+.3f}")

        if abs(diff) <= threshold:
            print(f"    bin {bin_index}: converged after {iteration} iteration(s).")
            return

        label_params["sholl_plot"]["mean"][bin_index] = max(MIN_MEAN, current + diff)
    else:
        print(f"    bin {bin_index}: did NOT converge within {max_iterations} iterations; keeping the last value.")


def calibrate_bifurcation_count(label_params, label, target_mean, threshold, n_generations, max_iterations,
                                 step_size, n_std, max_attempts_per_window, max_total_attempts, max_workers):
    """
    Shift label_params['bifurcation_count']['mean'] until the average
    simulated bifurcation count, across n_generations FRESH trees each
    grown fully (distance_limit=None) via synthesize_progressive, is
    within threshold of target_mean, or max_iterations is reached.
    std is never touched.
    """
    for iteration in range(1, max_iterations + 1):
        trees = build_and_grow_trees(
            label_params, label, n_generations, step_size, n_std, max_attempts_per_window,
            max_total_attempts, None, max_workers,
        )

        simulated = simulated_bifurcation_mean(trees)
        diff = target_mean - simulated
        current = label_params["bifurcation_count"]["mean"]

        print(f"    iteration {iteration}: working_mean={current:.3f} simulated={simulated:.3f} target={target_mean:.3f} diff={diff:+.3f}")

        if abs(diff) <= threshold:
            print(f"    converged after {iteration} iteration(s).")
            return

        label_params["bifurcation_count"]["mean"] = max(MIN_MEAN, current + diff)
    else:
        print(f"    did NOT converge within {max_iterations} iterations; keeping the last value.")


def calibrate(original_params, sholl_threshold, bifurcation_threshold, n_generations, max_iterations,
              step_size, n_std, max_attempts_per_window, max_total_attempts, max_workers):
    working_params = copy.deepcopy(original_params)

    for label in LABELS:
        if label not in working_params:
            continue

        bin_size = original_params[label]["bin_size"]
        n_bins = len(original_params[label]["sholl_plot"]["mean"])

        print(f"\n=== {label}: Sholl plot, bin by bin ===")
        for bin_index in range(n_bins):
            target_mean = original_params[label]["sholl_plot"]["mean"][bin_index]
            calibrate_sholl_bin(
                working_params[label], label, bin_index, bin_size, target_mean, sholl_threshold,
                n_generations, max_iterations, step_size, n_std, max_attempts_per_window, max_total_attempts, max_workers,
            )

        print(f"\n=== {label}: bifurcation count ===")
        target_mean = original_params[label]["bifurcation_count"]["mean"]
        calibrate_bifurcation_count(
            working_params[label], label, target_mean, bifurcation_threshold,
            n_generations, max_iterations, step_size, n_std, max_attempts_per_window, max_total_attempts, max_workers,
        )

    return working_params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parameters_json", help="Path to a preset's *.parameters.json file.")
    parser.add_argument("--output", default=None, help="Path to write the corrected JSON (default: <input>.corrected.json).")
    parser.add_argument("--sholl-threshold", type=float, default=0.5, help="Max allowed |experimental - simulated| for each Sholl bin before shifting (default: 0.5).")
    parser.add_argument("--bifurcation-threshold", type=float, default=0.5, help="Max allowed |experimental - simulated| for bifurcation count before shifting (default: 0.5).")
    parser.add_argument("--n-generations", type=int, default=200, help="Number of fresh trees synthesized per test (default: 200).")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max shift-and-retest iterations per bin/metric (default: 20).")
    parser.add_argument("--step-size", type=float, default=2, help="Synthesis step size (default: 2).")
    parser.add_argument("--n-std", type=float, default=3.0, help="Sholl/bifurcation-count tolerance passed to synthesize_progressive (default: 3.0).")
    parser.add_argument("--max-attempts-per-window", type=int, default=10, help="synthesize_progressive's own rollback-window retry limit (default: 10).")
    parser.add_argument("--max-total-attempts", type=int, default=1000, help="synthesize_progressive's own shared attempt budget (default: 1000).")
    parser.add_argument("--max-workers", type=int, default=None, help="Thread pool size per test (default: os.cpu_count()).")
    args = parser.parse_args()

    original_params = load_params(args.parameters_json)

    corrected = calibrate(
        original_params, args.sholl_threshold, args.bifurcation_threshold, args.n_generations, args.max_iterations,
        args.step_size, args.n_std, args.max_attempts_per_window, args.max_total_attempts, args.max_workers,
    )

    output_path = args.output or str(Path(args.parameters_json).with_suffix(".corrected.json"))
    Path(output_path).write_text(json.dumps(corrected, indent=2))
    print(f"\nwrote corrected parameters to {output_path}")


if __name__ == "__main__":
    main()
