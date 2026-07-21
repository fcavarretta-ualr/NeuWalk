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


##def _ids_by_section(index, parent_id):
##    # check the ids
##    tmp = [node_id for node_id, neurite in index.items() if neurite == index[parent_id]]
##    return min(tmp), max(tmp)
##
##
##def _tweak_neurite(index, parent_id, new_child):
##    parent_init_node_id, parent_last_node_id = _ids_by_section(index, parent_id)
##    
##    # get the parent
##    parent = index[parent_id]
##    
##    # divide children
##    local_cut_index = parent_id - parent_init_node_id
##    
##    # tweak the section
##    old_child = Neurite(points=parent.points[local_cut_index:].copy(), section_type=parent.section_type, parent=parent)
##
##    # cut the points
##    parent.points = parent.points[:local_cut_index+1]
##
##    # update the ids with the child
##    for node_id in range(parent_id + 1, parent_last_node_id + 1):
##        index[node_id] = old_child
##                
##    # connect previous children with the last section
##    for ch in parent.children:
##        if ch not in [old_child, new_child]:
##            ch.disconnect_from_parent()
##            ch.connect(old_child, relation="parent")
##
##
##        
##def read_swc(filename):
##    """
##    Read an SWC file and return its root Neurite objects.
##
##    SWC integer type codes are converted to string ``section_type`` labels
##    according to ``TYPE_LABELS``.
##
##    Returns
##    -------
##    list of Neurite
##        One root object for each disconnected tree in the SWC file.
##    """
##    data = np.loadtxt(filename, comments="#", ndmin=2)
##
##    if data.shape[1] != 7:
##        raise ValueError("An SWC file must contain exactly 7 columns.")
##
##    # organize the information
##    ids = data[:, 0].astype(int)
##    types = data[:, 1].astype(int)
##    points = data[:, 2:5].astype(float)
##    radii = data[:, 5].astype(float)
##    parent_ids = data[:, 6].astype(int)
##
##    if len(np.unique(ids)) != len(ids):
##        raise ValueError("SWC node identifiers must be unique.")
##
##    unknown_codes = sorted(set(types) - set(TYPE_LABELS))
##
##    if unknown_codes:
##        raise ValueError(
##            f"Unsupported SWC type code(s): {unknown_codes}."
##        )
##
##
##    # initialize the node_index
##    index = { i-1:None for i in ids}
##                
##    for node_id, parent_id in zip(ids, parent_ids):
##        node_id -= 1
##        parent_id -= 1
##            
##        if parent_id < 0:
##            index[node_id] = Neurite(section_type=TYPE_LABELS[types[node_id]])
##            
##        elif node_id > parent_id + 1:
##            # create the new section
##            index[node_id] = Neurite(section_type=TYPE_LABELS[types[node_id]], parent=index[parent_id])
##
##            # check the ids of the parent
##            parent_init_node_id, parent_last_node_id = _ids_by_section(index, parent_id)
##
##            # check boundaries
##            if not ( parent_init_node_id <= parent_id <= parent_last_node_id ):
##                raise ValueError("Inappropriate point id for parent")
##
##            # we need to cut this dendrite if parent_id is within the interval
##            _tweak_neurite(index, parent_id, index[node_id])
##            
##        elif types[node_id] != types[parent_id]:
##            # create new section and connect with parent
##            index[node_id] = Neurite(section_type=TYPE_LABELS[types[node_id]], parent=index[parent_id])
##            
##        else:
##            # copy the Neurite object
##            index[node_id] = index[parent_id]
##
##        # attach the point
##        index[node_id].points.append(np.array(points[node_id, :], dtype=float))
##
##
##
##    roots = [section for section in set(index.values()) if not section.parent]
##
##    # add first points to children
##    for root in roots:
##        for section in root.subtree:
##            if section.children:
##                first_point, last_point = section.points[0].copy(), section.points[-1].copy()            
##                for ch in section.children:
##                    first_distance, last_distance = np.linalg.norm(ch.points[0] - first_point), np.linalg.norm(ch.points[0] - last_point)
##                    
##                    if first_distance < last_distance:
##                        ch.points.insert(0, first_point)
##                    else:
##                        ch.points.insert(0, last_point)
##
##    return roots
##
##def write_swc(filename, roots, default_radius=1.0):
##    """Write a list of root section trees to an SWC file."""
##    if not isinstance(roots, (list, tuple)):
##        raise TypeError("roots must be a list or tuple.")
##    
##    if not roots:
##        raise ValueError("roots cannot be empty.")
##
##    if sum([not r.parent for r in roots]) != len(roots):
##        raise TypeError("roots should have no parent")
##
##    with open(filename, "w", encoding="utf-8") as file:
##        file.write("# id type x y z radius parent\n")
##        
##        index = {}
##        node_id = 0
##
##        # visit all the roots
##        for root in roots:
##            # subtree return the tree nodes in topological order
##            for section in root.subtree:
##
##                # retrieve node code
##                type_code = TYPE_CODES[section.section_type]
##
##                if section.parent:
##                    start_point = 1
##                else:
##                    start_point = 0
##
##                # get the points
##                for i, (x, y, z) in enumerate(section.points[start_point:]):
##
##                    # parent id
##                    if i == 0:
##                        # if it has no parent, -1, otherwise retrieve the parent id
##                        if not section.parent:
##                            parent_id = -1
##                        else:
##                            parent_id = index[section.parent]
##
##                    # same section
##                    else:
##                        parent_id = node_id
##                        
##                    node_id += 1
##
##                    # output the point
##                    file.write(f"{node_id} {type_code} {x:.9g} {y:.9g} {z:.9g} {default_radius} {parent_id}\n")
##
##                    
##                # memorize the section id
##                if section.has_children:
##                    index[section] = node_id
##                
##
##


def read_swc(filename):
    """Read an SWC file and return a list of root Neurite objects."""
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
        raise ValueError(f"Unsupported SWC type code(s): {unknown_codes}.")

    nodes = {
        node_id: {
            "type": node_type,
            "point": point,
            "radius": radius,
            "parent_id": parent_id,
            "children": [],
        }
        for node_id, node_type, point, radius, parent_id
        in zip(ids, types, points, radii, parent_ids)
    }

    for node_id, node in nodes.items():
        parent_id = node["parent_id"]

        if parent_id == -1:
            continue

        if parent_id not in nodes:
            raise ValueError(f"Node {node_id} references missing parent {parent_id}.")

        if parent_id == node_id:
            raise ValueError(f"Node {node_id} cannot be its own parent.")

        nodes[parent_id]["children"].append(node_id)

    root_ids = [node_id for node_id in ids if nodes[node_id]["parent_id"] == -1]

    if not root_ids:
        raise ValueError("The SWC file contains no root nodes.")

    def starts_new_section(node_id):
        parent_id = nodes[node_id]["parent_id"]

        if parent_id == -1:
            return True

        parent = nodes[parent_id]

        return (
            len(parent["children"]) != 1
            or len(nodes[node_id]["children"]) > 1
            or nodes[node_id]["type"] != parent["type"]
        )

    sections = {}
    node_to_section = {}

    def build_section(start_id, parent_section=None):
        node = nodes[start_id]
        section = Neurite(
            section_type=TYPE_LABELS[node["type"]],
            parent=parent_section,
        )

        if parent_section is not None:
            parent_id = node["parent_id"]
            section.points.append(nodes[parent_id]["point"].copy())

        current_id = start_id

        while True:
            current = nodes[current_id]
            section.points.append(current["point"].copy())
            node_to_section[current_id] = section

            children = current["children"]

            if len(children) != 1:
                break

            child_id = children[0]

            if nodes[child_id]["type"] != current["type"]:
                break

            current_id = child_id

        sections[start_id] = section

        for child_id in nodes[current_id]["children"]:
            build_section(child_id, section)

        return section

    roots = [build_section(root_id) for root_id in root_ids]
    return roots



def write_swc(filename, roots, default_radius=1.0):
    """Write a list of root section trees to an SWC file."""
    if not isinstance(roots, (list, tuple)):
        raise TypeError("roots must be a list or tuple.")

    if not roots:
        raise ValueError("roots cannot be empty.")

    if any(root.parent is not None for root in roots):
        raise ValueError("All roots must have no parent.")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# id type x y z radius parent\n")

        last_node_ids = {}
        node_id = 0

        for root in roots:
            for section in root.subtree:
                type_code = TYPE_CODES[section.section_type]

                if section.parent is None:
                    if len(section.points) < 1:
                        raise ValueError("A root section must contain at least one point.")
                    start_point = 0
                else:
                    if len(section.points) < 2:
                        raise ValueError("A child section must contain the shared point and at least one additional point.")
                    if section.parent not in last_node_ids:
                        raise ValueError("The parent section has not been written.")
                    start_point = 1

                parent_id = -1 if section.parent is None else last_node_ids[section.parent]

                for x, y, z in section.points[start_point:]:
                    node_id += 1
                    file.write(f"{node_id} {type_code} {x:.9g} {y:.9g} {z:.9g} {default_radius:.9g} {parent_id}\n")
                    parent_id = node_id

                last_node_ids[section] = node_id
