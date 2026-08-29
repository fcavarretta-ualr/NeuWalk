"""
Flag synthetic SWC reconstructions whose per-subsection total dendritic
length is an outlier relative to an experimental reference dataset.

For each subsection (apical_dendrite, basal_dendrite, apical_oblique,
unless skipped via --exclusion), this script:

1. Loads the EXPERIMENTAL reconstructions (experimental_dir) the same
   way extract_apc.py/extract_neo.py do -- via load_morphologies with
   that subsection's own delete_labels (deleting every other dendrite
   type, soma, axon, and secondary variants first, so what's left is
   only that subsection's own sections, merged/pruned by
   process_morphology the same way it always is) -- and computes the
   mean and std of total dendritic length across those files.

2. Loads the SYNTHETIC reconstructions (synthetic_dir) the same way,
   and computes each file's own total dendritic length for that
   subsection.

3. Any synthetic file whose length for ANY (not all) checked
   subsection falls outside experimental_mean +/- n_std*experimental_std
   is MOVED (not deleted) into --outlier-dir, so synthetic_dir ends up
   holding only reconstructions that fall within range on every
   checked subsection.

A file with no sections of a given subsection at all (e.g. a cell type
that genuinely never has obliques) contributes no data point for that
subsection, in either directory -- a bare zero would otherwise pull
that subsection's mean/std down in a way that doesn't reflect a real
measurement, and a zero-length synthetic file would look like an
outlier for the wrong reason.

Usage
-----
    python flag_length_outliers.py SYNTHETIC_DIR EXPERIMENTAL_DIR
    python flag_length_outliers.py SYNTHETIC_DIR EXPERIMENTAL_DIR --exclusion apical_oblique
    python flag_length_outliers.py SYNTHETIC_DIR EXPERIMENTAL_DIR --n-std 2.5 --outlier-dir SYNTHETIC_DIR/flagged --dry-run
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
    """
    morphologies = load_morphologies(directory, delete_labels=LABEL_DELETE_SETS[subsection], return_file_names=True)

    lengths = {}

    for filename, roots in morphologies:
        length = total_length(roots)
        if length > 0:
            lengths[filename] = length

    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("synthetic_dir", help="Directory of synthetic/generated .swc reconstructions to check.")
    parser.add_argument("experimental_dir", help="Directory of experimental .swc reconstructions used as the reference.")
    parser.add_argument("--exclusion", nargs="+", choices=SUBSECTIONS, default=[], metavar="SUBSECTION", help="Subsection(s) to skip checking entirely (choices: %(choices)s).")
    parser.add_argument("--n-std", type=float, default=3.0, help="Number of standard deviations defining the acceptable range (default: 3.0).")
    parser.add_argument("--outlier-dir", default=None, help="Directory to move outlier files into (default: <synthetic_dir>/outliers).")
    parser.add_argument("--dry-run", action="store_true", help="Report outliers without actually moving any files.")
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic_dir)
    experimental_dir = Path(args.experimental_dir)
    outlier_dir = Path(args.outlier_dir) if args.outlier_dir else synthetic_dir / "outliers"

    subsections = [s for s in SUBSECTIONS if s not in args.exclusion]
    if not subsections:
        raise ValueError("Every subsection was excluded via --exclusion; nothing to check.")

    print(f"Checking subsections: {', '.join(subsections)}")
    print()

    # 1. experimental mean/std of total length, per subsection
    experimental_stats = {}

    for subsection in subsections:
        lengths = load_lengths_per_file(experimental_dir, subsection)

        if not lengths:
            print(f"  {subsection}: no experimental reconstructions have this subsection -- skipping it entirely.")
            continue

        values = np.array(list(lengths.values()), dtype=float)
        experimental_stats[subsection] = (float(values.mean()), float(values.std()))
        print(f"  {subsection}: experimental total length mean={values.mean():.2f}  std={values.std():.2f}  (n={len(values)})")

    if not experimental_stats:
        print("\nNo experimental data found for any checked subsection; nothing to compare against.")
        return

    # 2. synthetic per-file, per-subsection total length
    synthetic_lengths = {subsection: load_lengths_per_file(synthetic_dir, subsection) for subsection in experimental_stats}

    # 3. an outlier is a file outside mean +/- n_std*std for ANY subsection
    outlier_reasons = {}  # filename -> list of (subsection, value, low, high)

    for subsection, (mean, std) in experimental_stats.items():
        low, high = mean - args.n_std * std, mean + args.n_std * std

        for filename, length in synthetic_lengths[subsection].items():
            if not (low <= length <= high):
                outlier_reasons.setdefault(filename, []).append((subsection, length, low, high))

    n_checked = len({filename for lengths in synthetic_lengths.values() for filename in lengths})

    print()
    print(f"Checked {n_checked} synthetic reconstruction(s); found {len(outlier_reasons)} outlier(s).")

    if not outlier_reasons:
        return

    if not args.dry_run:
        outlier_dir.mkdir(parents=True, exist_ok=True)

    for filename, reasons in sorted(outlier_reasons.items()):
        reason_str = "; ".join(f"{s}={v:.2f} not in [{lo:.2f}, {hi:.2f}]" for s, v, lo, hi in reasons)
        print(f"  {filename.name}: {reason_str}")

        if not args.dry_run:
            shutil.move(str(filename), str(outlier_dir / filename.name))

    if args.dry_run:
        print("\n--dry-run: no files were actually moved.")
    else:
        print(f"\nMoved {len(outlier_reasons)} file(s) to {outlier_dir}")


if __name__ == "__main__":
    main()
