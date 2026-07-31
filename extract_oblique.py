import argparse

from neuwalk.analysis.morphologies import load_morphologies


def iter_sections(section):
    yield section

    for child in section.children:
        yield from iter_sections(child)


parser = argparse.ArgumentParser(description="Load and retain only apical-oblique sections.")
parser.add_argument("directory", help="Directory containing SWC files.")
args = parser.parse_args()

morphologies = load_morphologies(args.directory, delete_labels="unknown")
apical_oblique_morphologies = []

for roots in morphologies:
    apical_obliques = []

    for root in roots:
        for section in iter_sections(root):
            if section.label != "apical_oblique":
                continue

            parent = section.parent

            if parent is None or parent.label != "apical_oblique":
                if parent is not None:
                    parent.disconnect(section)

                apical_obliques.append(section)

    apical_oblique_morphologies.append(apical_obliques)

print(f"Loaded {len(morphologies)} morphologies.")
print(f"Extracted {sum(map(len, apical_oblique_morphologies))} apical-oblique trees.")
