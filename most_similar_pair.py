"""
Compare every neuron in directory A against every neuron in directory
B, and report the most similar pair.

For each dendrite type (apical_dendrite, basal_dendrite,
apical_oblique), each neuron's Sholl plot, total length, and
bifurcation count are computed the same way extract_statistics does:
via load_morphologies with that type's own delete_labels (deleting
every other dendrite type, soma, axon, and secondary variants first),
with soma_processing=False -- process_soma would otherwise insert a
spurious point at the start of every orphaned root, which is harmless
for apical/basal (which attach near the real soma) but substantially
distorts apical_oblique (which attaches at scattered points along the
trunk); see extract_neo.py's own bug for the same issue.

For a given (neuron in A, neuron in B) pair, per dendrite type present
in both:

- Sholl plot difference: the two arrays are zero-padded to the same
  length (neurons reach different max radii), then the MEAN of the
  absolute per-bin differences is taken.
- Bifurcation count difference: |count_a - count_b|.
- Total length difference: |length_a - length_b|.

These are SUMMED across every dendrite type present in both neurons of
the pair (a type missing from either neuron is skipped for that pair,
not treated as zero) to get one sholl_score, bifurcation_score, and
length_score per pair.

Pairs are ranked by (sholl_score, bifurcation_score, length_score),
ascending -- i.e. primarily by Sholl-plot similarity, using
bifurcation count as a tiebreaker and total length as a further
tiebreaker. The most similar pair (lowest such tuple) is reported.

Usage
-----
    python most_similar_pair.py DIR_A DIR_B
    python most_similar_pair.py DIR_A DIR_B --bin-size 20 --top 5
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from neuwalk.analysis.morphologies import load_morphologies


LABELS = ("apical_dendrite", "basal_dendrite", "apical_oblique")

# Matches extract_apc.py/extract_neo.py's own discarded_sections exactly:
# for a given dendrite type's own statistics, every section belonging
# to a DIFFERENT dendritic lineage (plus soma, axon, unknown, and
# secondary oblique/dendrite variants) is deleted before measuring.
LABEL_DELETE_SETS = {
    "basal_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite", "axon"],
    "apical_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
    "apical_oblique": ["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
}


def total_length(roots):
    """Sum of section.length across every root's subtree."""
    return sum(section.length for root in roots for section in root.subtree)


def label_bifurcation_count(roots):
    """
    Bifurcation count across every root's subtree: a section with 2
    children only counts if BOTH children share the section's own
    label, matching extract_statistics' definition (a section whose
    children have different labels is an "internal bifurcation" there,
    excluded from bifurcation_count). After delete_labels, every
    remaining section already shares one label anyway, so this check
    is a no-op safety net rather than something that actually excludes
    anything here.
    """
    return sum(
        len(section.children) == 2 and all(child.label == section.label for child in section.children)
        for root in roots
        for section in root.subtree
    )


def label_sholl_plot(roots, bin_size):
    """
    Sholl-style crossing counts across every root's subtree: bin i
    counts how many sections pass through [i*bin_size, (i+1)*bin_size)
    at least once, using each section's full min-to-max distance range
    along its own path (not just individual segment direction), and
    counting a section at most once per bin even if its path revisits
    that bin.

    Distances are measured from the origin, not a soma point: with
    soma_processing=False and soma itself deleted, there's no soma
    point in this tree at all, but process_morphology's
    translate_sections already recenters every root at its own first
    point -- exactly the origin -- so every root already starts there.

    Returns
    -------
    numpy.ndarray
        Length is one more than the furthest bin any section reaches;
        an empty tree returns an all-zero array of length 1.
    """
    source = np.zeros(3)

    spans = []
    max_bin = 0

    for root in roots:
        for section in root.subtree:
            distances = np.linalg.norm(np.asarray(section.points, dtype=float) - source, axis=1)
            min_bin = int(distances.min() / bin_size)
            this_max_bin = int(distances.max() / bin_size)
            spans.append((min_bin, this_max_bin))
            max_bin = max(max_bin, this_max_bin)

    if not spans:
        return np.zeros(1, dtype=int)

    counts = np.zeros(max_bin + 1, dtype=int)

    for min_bin, this_max_bin in spans:
        counts[min_bin:this_max_bin + 1] += 1

    return counts


def load_neuron_data(directory, label, bin_size):
    """
    Load every .swc file in directory, processed for the given dendrite
    type, and return {filename: {"sholl_plot": array, "total_length":
    float, "bifurcation_count": int}}. A file with no sections of this
    type at all is omitted entirely -- not a real measurement of it,
    just its absence.
    """
    morphologies = load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[label], return_file_names=True, soma_processing=False)

    data = {}

    for filename, roots in morphologies:
        length = total_length(roots)

        if length == 0:
            continue

        data[filename] = {
            "sholl_plot": label_sholl_plot(roots, bin_size),
            "total_length": length,
            "bifurcation_count": label_bifurcation_count(roots),
        }

    return data


def sholl_difference(sholl_a, sholl_b):
    """Mean absolute per-bin difference between two Sholl arrays, zero-padded to the same length."""
    n = max(len(sholl_a), len(sholl_b))
    padded_a = np.pad(np.asarray(sholl_a, dtype=float), (0, n - len(sholl_a)))
    padded_b = np.pad(np.asarray(sholl_b, dtype=float), (0, n - len(sholl_b)))
    return float(np.mean(np.abs(padded_a - padded_b)))


def compare_all_pairs(dir_a, dir_b, bin_size):
    """
    Compare every neuron in dir_a against every neuron in dir_b.

    Returns a list of dicts (one per comparable pair), sorted ascending
    by (sholl_score, bifurcation_score, length_score) -- the first
    entry is the most similar pair.
    """
    data_a = {label: load_neuron_data(dir_a, label, bin_size) for label in LABELS}
    data_b = {label: load_neuron_data(dir_b, label, bin_size) for label in LABELS}

    files_a = sorted(set().union(*(d.keys() for d in data_a.values())), key=str)
    files_b = sorted(set().union(*(d.keys() for d in data_b.values())), key=str)

    results = []

    for file_a, file_b in product(files_a, files_b):
        sholl_score = 0.0
        bifurcation_score = 0.0
        length_score = 0.0
        labels_compared = []

        for label in LABELS:
            entry_a = data_a[label].get(file_a)
            entry_b = data_b[label].get(file_b)

            if entry_a is None or entry_b is None:
                continue

            sholl_score += sholl_difference(entry_a["sholl_plot"], entry_b["sholl_plot"])
            bifurcation_score += abs(entry_a["bifurcation_count"] - entry_b["bifurcation_count"])
            length_score += abs(entry_a["total_length"] - entry_b["total_length"])
            labels_compared.append(label)

        if not labels_compared:
            continue

        results.append({
            "file_a": file_a,
            "file_b": file_b,
            "sholl_score": sholl_score,
            "bifurcation_score": bifurcation_score,
            "length_score": length_score,
            "labels_compared": labels_compared,
        })

    results.sort(key=lambda r: (r["sholl_score"], r["bifurcation_score"], r["length_score"]))

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", help="First directory of .swc files.")
    parser.add_argument("dir_b", help="Second directory of .swc files.")
    parser.add_argument("--bin-size", type=float, default=50.0, help="Sholl bin size (default: 50.0).")
    parser.add_argument("--top", type=int, default=1, help="Show the top N most similar pairs (default: 1).")
    args = parser.parse_args()

    results = compare_all_pairs(Path(args.dir_a), Path(args.dir_b), args.bin_size)

    if not results:
        print("No comparable pairs found (no shared dendrite type between any neuron in A and any in B).")
        return

    print(f"Compared {len(results)} pair(s) across A x B.")
    print()

    for rank, r in enumerate(results[:args.top], start=1):
        print(f"#{rank}: A={r['file_a'].name}  B={r['file_b'].name}")
        print(f"    sholl_score={r['sholl_score']:.4f}  bifurcation_score={r['bifurcation_score']:.4f}  length_score={r['length_score']:.4f}")
        print(f"    dendrite types compared: {', '.join(r['labels_compared'])}")
        print()

    best = results[0]
    print(f"Most similar pair: {best['file_a']}  <->  {best['file_b']}")


if __name__ == "__main__":
    main()
