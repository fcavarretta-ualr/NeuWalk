<<<<<<< HEAD
#from neuwalk.presets.olfactory_bulb.mitral import generate
from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate
#from neuwalk.presets.anterior_piriform_cortex.semilunar import generate
#from neuwalk.presets.neocortex.pyramidal import generate
from neuwalk.presets._common import synthesize_topologies
import numpy as np
import json
from pathlib import Path

step_size = 0.5 #2.5
n_std = 3
path = Path("neuwalk/presets/anterior_piriform_cortex/pyramidal.parameters.json")
all_params = json.loads(path.read_text())
del all_params['basal_dendrite']
    
cnt = []
for seed in range(200):
  ret = synthesize_topologies(all_params, seed, step_size, n_std, 10, 1000, False, with_soma=None)['apical_dendrite']
  cnt.append(ret['topology'].soma.bifurcation_count)
  print(seed, cnt[-1], np.mean(cnt), np.std(cnt))


#soma = generate(seed, step_size=step_size)['output']
#print(ret['topology'].soma.bifurcation_count, sum(s.bifurcation_count for s in soma.children if s.label == "apical_dendrite"))
#print(np.mean(cnt), np.std(cnt))
=======
"""
Synthesize many topologies in parallel from a preset's *.parameters.json
file, then plot bifurcation_count and sholl_plot for apical_dendrite and
basal_dendrite, comparing the experimental targets (from the JSON) against
the simulated results (from actually running synthesis across many seeds).

Usage
-----
    python compare_topology_stats.py neuwalk/presets/anterior_piriform_cortex/semilunar.parameters.json
    python compare_topology_stats.py PARAMS.json --n-seeds 200 --max-workers 8
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuwalk.presets._common import synthesize_topologies


LABELS = ("apical_dendrite", "basal_dendrite", "apical_oblique")


def load_params(path):
    return json.loads(Path(path).read_text())


def synthesize_one(all_params, seed, step_size, n_std, max_attempts_per_window, max_total_attempts):
    """
    Synthesize topologies for one seed and extract bifurcation_count and
    sholl_plot for every label in LABELS that all_params provides. Runs
    in a worker thread; every synthesize_topologies call builds its own
    fresh Random(seed) internally, so concurrent calls don't share state.
    """
    ret = synthesize_topologies(
        all_params,
        seed,
        step_size,
        n_std,
        max_attempts_per_window,
        max_total_attempts,
        False,
        # apical_oblique (if present) is meant to be grafted onto
        # apical_dendrite, not stand alone with its own soma
        with_soma=lambda label: label != "apical_oblique",
    )

    result = {}

    for label in LABELS:
        if label not in ret:
            continue

        topology = ret[label]["topology"]
        result[label] = {
            "bifurcation_count": topology.bifurcation_count(),
            "sholl_plot": topology.sholl_plot(),
        }

    return result


def run_batch(all_params, n_seeds, step_size, n_std, max_attempts_per_window, max_total_attempts, max_workers=None):
    """Run n_seeds syntheses concurrently across a thread pool."""
    results = {label: {"bifurcation_count": [], "sholl_plot": []} for label in LABELS if label in all_params}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                synthesize_one, all_params, seed, step_size, n_std,
                max_attempts_per_window, max_total_attempts,
            ): seed
            for seed in range(n_seeds)
        }

        for future in as_completed(futures):
            seed = futures[future]

            try:
                one_result = future.result()
            except Exception as error:
                print(f"seed {seed} FAILED: {error}")
                continue

            for label, values in one_result.items():
                results[label]["bifurcation_count"].append(values["bifurcation_count"])
                results[label]["sholl_plot"].append(values["sholl_plot"])

            print(f"seed {seed} done {one_result['apical_dendrite']['bifurcation_count']}")

    return results


def pad_to_common_length(arrays):
    """Pad a list of 1D arrays with zeros to a common (max) length."""
    max_len = max(len(a) for a in arrays)
    return np.array([np.pad(np.asarray(a, dtype=float), (0, max_len - len(a))) for a in arrays])


def plot_bifurcation_count(results, all_params, n_experimental_samples, rng, output_path):
    """
    Boxplots of bifurcation_count, experimental (gray) vs. simulated
    (black), one subplot per label.

    The JSON only stores the experimental mean and std for
    bifurcation_count, not the original per-neuron values, so the
    experimental boxplot is drawn from a Normal(mean, std) sample rather
    than genuine experimental data -- it approximates the spread implied
    by those two numbers, it is not the real distribution.
    """
    labels = list(results)
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), squeeze=False)
    axes = axes[0]

    for ax, label in zip(axes, labels):
        simulated = np.asarray(results[label]["bifurcation_count"], dtype=float)

        print(label, 'mean=', np.mean(simulated), 'SD=', np.std(simulated))

        exp_stats = all_params[label]["bifurcation_count"]
        experimental = rng.normal(exp_stats["mean"], exp_stats["std"], size=n_experimental_samples)

        bp = ax.boxplot(
            [experimental, simulated],
            tick_labels=["experimental", "simulated"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("lightgray")
        bp["boxes"][0].set_edgecolor("gray")
        bp["medians"][0].set_color("dimgray")
        bp["boxes"][1].set_facecolor("black")
        bp["boxes"][1].set_edgecolor("black")
        bp["medians"][1].set_color("white")

        ax.set_title(label)
        ax.set_ylabel("bifurcation count")

    fig.suptitle(
        "Bifurcation count: experimental (gray) vs. simulated (black)\n"
        "experimental drawn from Normal(mean, std); only summary statistics are available"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sholl(results, all_params, output_path):
    """
    Sholl-plot crossings by radial bin, mean +/- std, experimental (gray)
    vs. simulated (black), one subplot per label.
    """
    labels = list(results)
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
    axes = axes[0]

    for ax, label in zip(axes, labels):
        bin_size = all_params[label]["bin_size"]

        simulated = pad_to_common_length(results[label]["sholl_plot"])
        sim_mean = simulated.mean(axis=0)
        sim_std = simulated.std(axis=0)
        sim_bins = np.arange(len(sim_mean)) * bin_size

        exp_mean = np.asarray(all_params[label]["sholl_plot"]["mean"], dtype=float)
        exp_std = np.asarray(all_params[label]["sholl_plot"]["std"], dtype=float)
        exp_bins = np.arange(len(exp_mean)) * bin_size

        ax.plot(exp_bins, exp_mean, color="gray", label="experimental")
        ax.fill_between(exp_bins, exp_mean - exp_std, exp_mean + exp_std, color="gray", alpha=0.3)

        ax.plot(sim_bins, sim_mean, color="black", label="simulated")
        ax.fill_between(sim_bins, sim_mean - sim_std, sim_mean + sim_std, color="black", alpha=0.3)

        ax.set_title(label)
        ax.set_xlabel("distance from soma")
        ax.set_ylabel("crossings")
        ax.legend()

    fig.suptitle("Sholl plot: experimental (gray) vs. simulated (black), mean \u00b1 std")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parameters_json", help="Path to a preset's *.parameters.json file.")
    parser.add_argument("--n-seeds", type=int, default=200, help="Number of seeds to synthesize (default: 200).")
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--n-std", type=float, default=3.0)
    parser.add_argument("--max-attempts-per-window", type=int, default=25)
    parser.add_argument("--max-total-attempts", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=None, help="Thread pool size (default: os.cpu_count()).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the experimental Normal-sample overlay.")
    parser.add_argument("--bifurcation-plot", default="bifurcation_count.png")
    parser.add_argument("--sholl-plot", default="sholl_plot.png")
    args = parser.parse_args()

    all_params = load_params(args.parameters_json)

    for params in all_params.values():
        if 'bifurcation_internal_density' in params:
            del params['bifurcation_internal_density']

    results = run_batch(
        all_params, args.n_seeds, args.step_size, args.n_std,
        args.max_attempts_per_window, args.max_total_attempts, args.max_workers,
    )

    rng = np.random.default_rng(args.seed)
    plot_bifurcation_count(results, all_params, args.n_seeds, rng, args.bifurcation_plot)
    plot_sholl(results, all_params, args.sholl_plot)

    print(f"saved {args.bifurcation_plot} and {args.sholl_plot}")


if __name__ == "__main__":
    main()
>>>>>>> 21a7a56 (last version)
