"""
Read all SWC files from directory A and directory B.

From directory B, estimate the mean and standard deviation of total
dendritic length for basal_dendrite, apical_dendrite, and
apical_oblique (if any reconstructions in B have it).

Then move the SWC files in directory A whose total length for any of
those subsections falls outside mean +/- n_std*std (using B's mean/std)
into a subdirectory bak.

Both directories are processed the same way extract_apc.py/extract_neo.py
do: via load_morphologies with each subsection's own delete_labels
(deleting every other dendrite type, soma, axon, and secondary variants
first, so what's left is only that subsection's own sections,
merged/pruned by process_morphology the same way it always is).

A file with no sections of a given subsection at all (e.g. a cell type
that genuinely never has obliques) contributes no data point for that
subsection, in either directory -- a bare zero would otherwise pull
B's mean/std down in a way that doesn't reflect a real measurement, and
would make a zero-length file in A look like an outlier for the wrong
reason.

Usage
-----
    python move_length_outliers.py DIR_A DIR_B
    python move_length_outliers.py DIR_A DIR_B --n-std 2.5
    python move_length_outliers.py DIR_A DIR_B --bak-dir DIR_A/bak --dry-run
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from neuwalk.analysis.morphologies import load_morphologies


SUBSECTIONS = ("apical_dendrite", "basal_dendrite", "apical_oblique")

# Matches extract_apc.py/extract_neo.py's own discarded_sections exactly:
# for a given subsection's own statistics, every section belonging to a
# DIFFERENT dendritic lineage (plus soma, axon, unknown, and secondary
# oblique/dendrite variants) is deleted before measuring -- what remains
# is only that subsection's own, merged/pruned sections.
LABEL_DELETE_SETS = {
    "basal_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "apical_dendrite", "axon"],
    "apical_dendrite": ["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
    "apical_oblique": ["unknown", "apical_dendrite", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"],
}


def total_length(roots):
    """Sum of section.length across every root's subtree."""
    return sum(section.length for root in roots for section in root.subtree)


def load_lengths_per_file(directory, subsection):
    """
    Load every .swc file in directory, processed for the given
    subsection (via load_morphologies with that subsection's own
    delete_labels), and return {filename: total_length}. Files with no
    sections of this subsection at all (total_length == 0) are omitted
    -- not a real measurement of this subsection, just its absence.

    soma_processing=False: "soma" is already in every subsection's own
    delete_labels above, so there's nothing left to recenter around --
    and when a file has none of this subsection at all, roots ends up
    empty after deletion, so process_soma's own attempt to average an
    empty list of points raises (a real gap in that function, but one
    this script doesn't need to touch, since it never needs a
    recentered soma point -- only section.length, which is unaffected
    by translation).
    """
    morphologies = load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[subsection], return_file_names=True, soma_processing=False)

    lengths = {}

    for filename, roots in morphologies:
        length = total_length(roots)
        if length > 0:
            lengths[filename] = length

    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", help="Directory of .swc files to check and, if needed, move files out of.")
    parser.add_argument("dir_b", help="Directory of .swc files used as the reference for mean/std.")
    parser.add_argument("--n-std", type=float, default=3.0, help="Number of standard deviations defining the acceptable range (default: 3.0).")
    parser.add_argument("--bak-dir", default=None, help="Directory to move outlier files into (default: <dir_a>/bak).")
    parser.add_argument("--dry-run", action="store_true", help="Report outliers without actually moving any files.")
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    bak_dir = Path(args.bak_dir) if args.bak_dir else dir_a / "bak"

    print("Reference statistics from directory B:")
    print()

    # 1. B's mean/std of total length, per subsection
    reference_stats = {}

    for subsection in SUBSECTIONS:
        lengths = load_lengths_per_file(dir_b, subsection)

        if not lengths:
            print(f"  {subsection}: no reconstructions in B have this subsection -- skipping it.")
            continue

        values = np.array(list(lengths.values()), dtype=float)
        reference_stats[subsection] = (float(values.mean()), float(values.std()))
        print(f"  {subsection}: mean={values.mean():.2f}  std={values.std():.2f}  (n={len(values)})")

    if not reference_stats:
        print("\nNo data found in B for any subsection; nothing to compare against.")
        return

    # 2. A's per-file, per-subsection total length
    a_lengths = {subsection: load_lengths_per_file(dir_a, subsection) for subsection in reference_stats}

    # 3. an outlier is a file outside mean +/- n_std*std for ANY subsection
    outlier_reasons = {}  # filename -> list of (subsection, value, low, high)

    for subsection, (mean, std) in reference_stats.items():
        low, high = mean - args.n_std * std, mean + args.n_std * std

        for filename, length in a_lengths[subsection].items():
            if not (low <= length <= high):
                outlier_reasons.setdefault(filename, []).append((subsection, length, low, high))

    n_checked = len({filename for lengths in a_lengths.values() for filename in lengths})

    print()
    print(f"Checked {n_checked} reconstruction(s) in A; found {len(outlier_reasons)} outlier(s).")

    if not outlier_reasons:
        return

    if not args.dry_run:
        bak_dir.mkdir(parents=True, exist_ok=True)

    for filename, reasons in sorted(outlier_reasons.items()):
        reason_str = "; ".join(f"{s}={v:.2f} not in [{lo:.2f}, {hi:.2f}]" for s, v, lo, hi in reasons)
        print(f"  {filename.name}: {reason_str}")

        if not args.dry_run:
            shutil.move(str(filename), str(bak_dir / filename.name))

    if args.dry_run:
        print("\n--dry-run: no files were actually moved.")
    else:
        print(f"\nMoved {len(outlier_reasons)} file(s) to {bak_dir}")


if __name__ == "__main__":
    main()
