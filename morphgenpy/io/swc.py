import numpy as np

from ..core.neurite import Neurite


TYPE_LABELS = {
    0: "unknown",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "apical_oblique",
    6: "apical_secondary_dendrite",
    7: "apical_secondary_oblique",
}

TYPE_CODES = {
    label: code
    for code, label in TYPE_LABELS.items()
}


def read_swc(filename):
    """
    Read an SWC file and return its root Neurite objects.

    SWC integer type codes are converted to string ``section_type`` labels
    according to ``TYPE_LABELS``.

    Returns
    -------
    list of Neurite
        One root object for each disconnected tree in the SWC file.
    """
    data = np.loadtxt(filename, comments="#", ndmin=2)

    if data.shape[1] != 7:
        raise ValueError("An SWC file must contain exactly 7 columns.")

    ids = data[:, 0].astype(int)
    types = data[:, 1].astype(int)
    points = data[:, 2:5].astype(float)
    radii = data[:, 5].astype(float)
    parent_ids = data[:, 6].astype(int)

    if len(np.unique(ids)) != len(ids):
        raise ValueError("SWC node identifiers must be unique.")

    unknown_codes = sorted(set(types) - set(TYPE_LABELS))

    if unknown_codes:
        raise ValueError(
            f"Unsupported SWC type code(s): {unknown_codes}."
        )

    index = {
        node_id: i
        for i, node_id in enumerate(ids)
    }
    children = {
        node_id: []
        for node_id in ids
    }

    roots = []

    for node_id, parent_id in zip(ids, parent_ids):
        if parent_id == -1:
            roots.append(node_id)

        elif parent_id not in index:
            raise ValueError(
                f"Node {node_id} references missing parent "
                f"{parent_id}."
            )

        else:
            children[parent_id].append(node_id)

    if not roots:
        raise ValueError("The SWC file contains no root nodes.")

    def make_section(
        start_id,
        parent_section=None,
        include_parent=False,
    ):
        section_ids = []

        if include_parent:
            section_ids.append(
                parent_ids[index[start_id]]
            )

        current_id = start_id
        section_ids.append(current_id)

        section_code = types[index[current_id]]
        section_type = TYPE_LABELS[section_code]

        while True:
            child_ids = children[current_id]

            if len(child_ids) != 1:
                break

            child_id = child_ids[0]

            if types[index[child_id]] != section_code:
                break

            current_id = child_id
            section_ids.append(current_id)

        section = Neurite(
            points=np.asarray(
                [
                    points[index[node_id]]
                    for node_id in section_ids
                ]
            ),
            section_type=section_type,
            parent=parent_section,
        )

        section.radii = np.asarray(
            [
                radii[index[node_id]]
                for node_id in section_ids
            ],
            dtype=float,
        )

        for child_id in children[current_id]:
            make_section(
                child_id,
                parent_section=section,
                include_parent=True,
            )

        if len(children[current_id]) == 1:
            child_id = children[current_id][0]

            if types[index[child_id]] != section_code:
                make_section(
                    child_id,
                    parent_section=section,
                    include_parent=True,
                )

        return section

    return [
        make_section(root_id)
        for root_id in roots
    ]


def write_swc(filename, roots, default_radius=1.0):
    """Write a list of root section trees to an SWC file."""
    if not isinstance(roots, (list, tuple)):
        raise TypeError("roots must be a list or tuple.")
    if not roots:
        raise ValueError("roots cannot be empty.")

    default_radius = float(default_radius)
    if not np.isfinite(default_radius) or default_radius <= 0.0:
        raise ValueError("default_radius must be finite and positive.")

    rows = []
    next_id = 1
    endpoint_ids = {}
    visited = set()

    def add_section(section):
        nonlocal next_id
        section_key = id(section)
        if section_key in visited:
            raise ValueError("A section is referenced more than once.")
        visited.add(section_key)

        points = np.asarray(section.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("section.points must have shape (n, 3).")
        if len(points) == 0:
            raise ValueError("Cannot write a section with no points.")
        if not np.all(np.isfinite(points)):
            raise ValueError("section.points must contain finite values.")

        section_type = "unknown" if section.section_type is None else section.section_type
        if section_type not in TYPE_CODES:
            raise ValueError(f"Unsupported section_type: {section_type!r}.")
        section_code = TYPE_CODES[section_type]

        start = 0
        parent_id = -1
        if section.parent is not None:
            parent_key = id(section.parent)
            if parent_key not in endpoint_ids:
                raise ValueError("The parent section must be written before its child.")
            parent_id = endpoint_ids[parent_key]
            parent_endpoint = np.asarray(section.parent.points[-1], dtype=float)
            if np.allclose(points[0], parent_endpoint):
                start = 1

        previous_id = parent_id
        for point in points[start:]:
            node_id = next_id
            next_id += 1
            x, y, z = point
            rows.append((node_id, section_code, x, y, z, default_radius, previous_id))
            previous_id = node_id

        endpoint_ids[section_key] = parent_id if start == len(points) else previous_id

        for child in section.children:
            if child.parent is not section:
                raise ValueError("Each child.parent must reference its parent section.")
            add_section(child)

    for root in roots:
        if root.parent is not None:
            raise ValueError("Each root section must have parent=None.")
        add_section(root)

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# id type x y z radius parent\n")
        for row in rows:
            file.write(
                f"{row[0]} {row[1]} {row[2]:.9g} {row[3]:.9g} "
                f"{row[4]:.9g} {row[5]:.9g} {row[6]}\n"
            )
