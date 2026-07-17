from pathlib import Path

import numpy as np

from .. import misc
from ..io import read_swc


def _normalize_section_types(section_types):
    if section_types is None:
        return set()

    return {section_types} if isinstance(section_types, str) else set(section_types)


def _align_children(section):
    """Translate each child so that it begins at its parent's endpoint."""
    for child in section.children:
        child.points = misc.translate_points(
            child.points,
            source=child.points[0],
            target=section.points[-1],
        )
        _align_children(child)


def _preprocess_geometry(root):
    """Center each root and connect every child geometrically to its parent."""
    if root.section_type == "soma":
        root.points = [np.zeros(3, dtype=float)]
    elif len(root.points):
        root.points = misc.translate_points(
            root.points,
            source=root.points[0],
        )

    _align_children(root)


def _remove_consecutive_duplicate_points(root):
    """Remove consecutive duplicate points from every section in the tree."""
    for section in root.wholetree:
        if len(section.points) < 2:
            continue

        points = [section.points[0]]

        for point in section.points[1:]:
            if not (point == points[-1]).all():
                points.append(point)

        section.points = points

def _delete_section_types(section, section_types):
    for child in list(section.children):
        _delete_section_types(child, section_types)

        if child.section_type in section_types:
            grandchildren = list(child.children)
            section.disconnect(child)

            for grandchild in grandchildren:
                child.disconnect(grandchild)
                section.connect(grandchild, relation="child")


def _merge_single_children(section):
    while (
        len(section.children) == 1
        and section.section_type == section.children[0].section_type
    ):
        section._merge_with_descendant()

    for child in list(section.children):
        _merge_single_children(child)


def process_morphology(
    roots,
    delete_section_types=None,
    merge_single_children=True,
):
    """Delete selected section types and optionally merge same-type single-child sections."""
    roots = list(roots)
    delete_section_types = _normalize_section_types(delete_section_types)

    for root in roots:
        _delete_section_types(root, delete_section_types)

    processed_roots = []

    for root in roots:
        if root.section_type not in delete_section_types:
            processed_roots.append(root)
            continue

        children = list(root.children)

        for child in children:
            root.disconnect(child)

        processed_roots.extend(children)

    if merge_single_children:
        for root in processed_roots:
            _merge_single_children(root)

    # Deleting or merging sections may reconnect descendants to new parents.
    for root in processed_roots:
        _align_children(root)

    return processed_roots

def _remove_single_point_sections(root):
    """Delete one-point sections and reconnect their descendants."""

    def process(section):
        replacements = []

        for child in list(section.children):
            section.disconnect(child)
            replacements.extend(process(child))

        if len(section.points) == 1 and section.section_type != "soma":
            return replacements

        for replacement in replacements:
            section.connect(replacement, relation="child")

        return [section]

    return process(root)

def load_morphologies(
    directory,
    root_section_types=None,
    delete_section_types="unknown",
    merge_single_children=True,
):
    """Load and process morphologies from all SWC files in a directory."""
    files = sorted(Path(directory).glob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

    root_section_types = (
        _normalize_section_types(root_section_types)
        if root_section_types is not None
        else None
    )

    morphologies = []

    for filename in files:
        roots = []

        for root in read_swc(filename):
            _preprocess_geometry(root)
            
            _remove_consecutive_duplicate_points(root)

            _remove_single_point_sections(root)
            
            candidates = (
                root.children
                if root.section_type == "soma"
                else [root]
            )

            roots.extend(
                process_morphology(
                    candidates,
                    delete_section_types,
                    merge_single_children,
                )
            )

        if root_section_types is not None:
            roots = [
                root
                for root in roots
                if root.section_type in root_section_types
            ]

        morphologies.append(roots)

    return morphologies
