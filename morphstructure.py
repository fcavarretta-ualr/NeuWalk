#!/usr/bin/env python3

import argparse
from collections import defaultdict

from morphgenpy.io.swc import read_swc



def print_hierarchy(roots):
    def print_section(section, prefix="", is_last=True, is_root=False):
        if len(section.points) > 2:
            first_point = section.points[1].tolist()
            last_point = section.points[-1].tolist()

            label = (
                f"{section.section_type} {section.length}\t"
                f"{tuple(first_point)} -> {tuple(last_point)}"
            )
        else:
            only_point = section.points[-1].tolist()
            label = (
                f"{section.section_type} {section.length}\t"
                f"{tuple(only_point)}"
            )            

        if is_root:
            print(label)
        else:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + label)

        children = section.children or []
        child_prefix = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(children):
            print_section(
                child,
                prefix="" if is_root else child_prefix,
                is_last=i == len(children) - 1,
            )

    for i, root in enumerate(roots):
        if i:
            print()

        print_section(root, is_root=True)


def main():
    parser = argparse.ArgumentParser(
        description="Print the hierarchical structure of an SWC morphology."
    )
    parser.add_argument("swc_file", help="Path to the SWC file")
    args = parser.parse_args()

    roots = read_swc(args.swc_file)
    print_hierarchy(roots)


if __name__ == "__main__":
    main()
