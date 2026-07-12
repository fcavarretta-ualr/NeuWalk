import numpy as np

from ..core.neurite import Neurite


def read_swc(filename):
    """
    Read an SWC file and return its root Neurite objects.

    Notes
    -----
    Each unbranched SWC chain becomes one Neurite section. Child sections
    include the shared branch point as their first point. SWC radii are stored
    in the optional ``radii`` attribute of each Neurite object.

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

    index = {node_id: i for i, node_id in enumerate(ids)}
    children = {node_id: [] for node_id in ids}

    roots = []
    for node_id, parent_id in zip(ids, parent_ids):
        if parent_id == -1:
            roots.append(node_id)
        elif parent_id not in index:
            raise ValueError(
                f"Node {node_id} references missing parent {parent_id}."
            )
        else:
            children[parent_id].append(node_id)

    if not roots:
        raise ValueError("The SWC file contains no root nodes.")

    def make_section(start_id, parent_section=None, include_parent=False):
        section_ids = []

        if include_parent:
            section_ids.append(parent_ids[index[start_id]])

        current_id = start_id
        section_ids.append(current_id)
        section_type = types[index[current_id]]

        while True:
            child_ids = children[current_id]

            if len(child_ids) != 1:
                break

            child_id = child_ids[0]

            if types[index[child_id]] != section_type:
                break

            current_id = child_id
            section_ids.append(current_id)

        section = Neurite(
            points=np.asarray(
                [points[index[node_id]] for node_id in section_ids]
            ),
            section_type=int(section_type),
            parent=parent_section,
        )

        section.radii = np.asarray(
            [radii[index[node_id]] for node_id in section_ids],
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

            if types[index[child_id]] != section_type:
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


def write_swc(filename, neurite, default_radius=1.0):
    """
    Write a Neurite tree to an SWC file.

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
        raise TypeError("neurite must be a Neurite object.")

    if default_radius <= 0:
        raise ValueError("default_radius must be positive.")

    rows = []
    next_id = 1
    endpoint_ids = {}

    def add_section(section):
        nonlocal next_id

        if len(section.points) == 0:
            raise ValueError("Cannot write a neurite with no points.")

        section_type = section.section_type

        if section_type is None:
            section_type = 3

        if not isinstance(section_type, (int, np.integer)):
            raise TypeError("section_type must be an integer SWC type.")

        radii = getattr(section, "radii", None)

        if radii is None:
            radii = np.full(len(section.points), default_radius, dtype=float)
        else:
            radii = np.asarray(radii, dtype=float)

            if radii.shape != (len(section.points),):
                raise ValueError(
                    "radii must have one value for each section point."
                )

            if np.any(radii <= 0):
                raise ValueError("All radii must be positive.")

        start = 0
        parent_id = -1

        if section.parent is not None:
            parent_id = endpoint_ids[id(section.parent)]

            if np.allclose(
                section.points[0],
                section.parent.points[-1],
            ):
                start = 1

        previous_id = parent_id

        for i in range(start, len(section.points)):
            node_id = next_id
            next_id += 1

            x, y, z = section.points[i]

            rows.append(
                (
                    node_id,
                    int(section_type),
                    x,
                    y,
                    z,
                    radii[i],
                    previous_id,
                )
            )

            previous_id = node_id

        if start == len(section.points):
            endpoint_ids[id(section)] = parent_id
        else:
            endpoint_ids[id(section)] = previous_id

        for child in section.children:
            add_section(child)

    add_section(neurite.root)

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# id type x y z radius parent\n")

        for row in rows:
            file.write(
                f"{row[0]} {row[1]} "
                f"{row[2]:.9g} {row[3]:.9g} {row[4]:.9g} "
                f"{row[5]:.9g} {row[6]}\n"
            )
