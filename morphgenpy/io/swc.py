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


def write_swc(
    filename,
    neurite,
    default_radius=1.0,
):
    """
    Write a Neurite tree to an SWC file.

    String ``section_type`` labels are converted to SWC integer codes
    according to ``TYPE_CODES``.

    Parameters
    ----------
    filename : str or path-like
        Output SWC filename.
    neurite : Neurite
        Any section belonging to the tree.
    default_radius : float, default 1.0
        Radius used when a section has no ``radii`` attribute.
    """
    if not isinstance(neurite, Neurite):
        raise TypeError(
            "neurite must be a Neurite object."
        )

    if default_radius <= 0:
        raise ValueError(
            "default_radius must be positive."
        )

    rows = []
    next_id = 1
    endpoint_ids = {}

    def add_section(section):
        nonlocal next_id

        if len(section.points) == 0:
            raise ValueError(
                "Cannot write a neurite with no points."
            )

        section_type = section.section_type

        if section_type is None:
            section_type = "unknown"

        if section_type not in TYPE_CODES:
            raise ValueError(
                f"Unsupported section_type: "
                f"{section_type!r}. "
                f"Expected one of {tuple(TYPE_CODES)}."
            )

        section_code = TYPE_CODES[section_type]
        radii = getattr(section, "radii", None)

        if radii is None:
            radii = np.full(
                len(section.points),
                default_radius,
                dtype=float,
            )

        else:
            radii = np.asarray(
                radii,
                dtype=float,
            )

            if radii.shape != (
                len(section.points),
            ):
                raise ValueError(
                    "radii must have one value for each "
                    "section point."
                )

            if np.any(radii <= 0):
                raise ValueError(
                    "All radii must be positive."
                )

        start = 0
        parent_id = -1

        if section.parent is not None:
            parent_id = endpoint_ids[
                id(section.parent)
            ]

            if np.allclose(
                section.points[0],
                section.parent.points[-1],
            ):
                start = 1

        previous_id = parent_id

        for i in range(
            start,
            len(section.points),
        ):
            node_id = next_id
            next_id += 1

            x, y, z = section.points[i]

            rows.append(
                (
                    node_id,
                    section_code,
                    x,
                    y,
                    z,
                    radii[i],
                    previous_id,
                )
            )

            previous_id = node_id

        endpoint_ids[id(section)] = (
            parent_id
            if start == len(section.points)
            else previous_id
        )

        for child in section.children:
            add_section(child)

    add_section(neurite.root)

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# id type x y z radius parent\n"
        )

        for row in rows:
            file.write(
                f"{row[0]} {row[1]} "
                f"{row[2]:.9g} "
                f"{row[3]:.9g} "
                f"{row[4]:.9g} "
                f"{row[5]:.9g} "
                f"{row[6]}\n"
            )
