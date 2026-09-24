#!/usr/bin/env python3
"""Extract synthesis parameters per dendritic section type.

For each requested group the morphologies are loaded with every other known
section type deleted, so the resulting statistics describe that group in
isolation.  Groups, discarded labels and the output path are all given on the
command line, so the script also serves cell types with a different set of
section labels -- for instance neurons without oblique dendrites.

The delete list is derived rather than written out: it is every label the
script knows about (the groups plus --discard) minus the ones the current
group keeps.  A label therefore has to be named somewhere, otherwise it is
silently retained in every extraction.

Run with no options to reproduce the original neocortical-pyramidal preset.
"""

import argparse
import json
import sys
from pathlib import Path

from neuwalk.analysis.morphologies import load_morphologies
from neuwalk.analysis.extraction import extract_statistics

DEFAULT_GROUPS = ["basal_dendrite", "apical_dendrite", "apical_oblique"]
DEFAULT_DISCARD = ["unknown", "soma", "axon",
                   "apical_secondary_dendrite", "apical_secondary_oblique"]
DEFAULT_DROP = ["total_length", "bifurcation_internal_density"]
# bifurcation_internal_density describes where obliques branch off the trunk,
# so it is measured on trunk and obliques together and stored on the trunk.
DEFAULT_DENSITY_FROM = ["apical_dendrite", "apical_oblique"]
DEFAULT_DENSITY_INTO = "apical_dendrite"


def parse_group(spec):
    """'name=lab1,lab2' or plain 'label' -> (name, [labels to keep])."""
    if "=" in spec:
        name, labels = spec.split("=", 1)
        keep = [lab.strip() for lab in labels.split(",") if lab.strip()]
        if not name.strip() or not keep:
            raise argparse.ArgumentTypeError(f"malformed group: {spec!r}")
        return name.strip(), keep
    return spec, [spec]


def extract(directory, keep, universe, bin_size):
    """Statistics over morphologies reduced to the labels in `keep`."""
    delete = sorted(universe - set(keep))
    return extract_statistics(
        load_morphologies(directory, delete_labels=delete, soma_processing=False),
        bin_size=bin_size)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bin-size", type=float, default=50.0)
    parser.add_argument(
        "-g", "--group", action="append", metavar="NAME[=LABEL,...]",
        help="section type to extract in isolation; repeat for several. "
             "NAME=LAB1,LAB2 keeps those labels together under NAME. "
             "Default: " + " ".join(DEFAULT_GROUPS))
    parser.add_argument(
        "-d", "--discard", nargs="*", metavar="LABEL",
        help="labels never retained by any group. "
             "Default: " + " ".join(DEFAULT_DISCARD))
    parser.add_argument(
        "--drop-param", nargs="*", metavar="KEY",
        help="statistics removed from the output (not used for synthesis). "
             "Default: " + " ".join(DEFAULT_DROP))
    parser.add_argument(
        "--density-from", nargs="+", metavar="LABEL",
        help="labels kept when measuring bifurcation_internal_density. "
             "Default: " + " ".join(DEFAULT_DENSITY_FROM))
    parser.add_argument(
        "--density-into", metavar="NAME",
        help="group that receives bifurcation_internal_density. "
             "Default: " + DEFAULT_DENSITY_INTO)
    parser.add_argument(
        "--no-density", action="store_true",
        help="skip bifurcation_internal_density entirely (cell types without obliques)")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("neuwalk/presets/neocortex/pyramidal.parameters.json"))
    args = parser.parse_args()

    groups = [parse_group(spec) for spec in (args.group or DEFAULT_GROUPS)]
    discard = DEFAULT_DISCARD if args.discard is None else args.discard
    drop = DEFAULT_DROP if args.drop_param is None else args.drop_param
    density_from = args.density_from or DEFAULT_DENSITY_FROM
    density_into = args.density_into or DEFAULT_DENSITY_INTO

    names = [name for name, _ in groups]
    if len(set(names)) != len(names):
        sys.exit("duplicate group name")

    # every label the script knows about; anything a group does not keep is deleted
    universe = set(discard)
    for _, keep in groups:
        universe |= set(keep)
    if not args.no_density:
        universe |= set(density_from)

    all_params = {}
    for name, keep in groups:
        print(f"Elaboration of {name}")
        try:
            print("\tExtracting statistics...", end="", flush=True)
            params = extract(args.directory, keep, universe, args.bin_size)
            print("done")
        except ValueError as err:
            print(f"skipped ({err})")
            continue
        for key in drop:
            params.pop(key, None)
        params["bin_size"] = args.bin_size
        all_params[name] = params

    if not all_params:
        sys.exit("no group could be extracted")

    if not args.no_density:
        if density_into not in all_params:
            sys.exit(f"cannot store bifurcation_internal_density: "
                     f"group {density_into!r} was not extracted")
        print(f"Elaboration of bifurcation_internal_density -> {density_into}")
        stats = extract(args.directory, density_from, universe, args.bin_size)
        all_params[density_into]["bifurcation_internal_density"] = \
            stats["bifurcation_internal_density"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(all_params, handle, indent=4, default=lambda value: value.tolist())
    print(f"Written {args.output}")


if __name__ == "__main__":
    main()
