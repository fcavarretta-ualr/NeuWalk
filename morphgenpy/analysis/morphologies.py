from pathlib import Path

import numpy as np

from .. import misc
from ..io import read_swc


def repair_single_point_section(roots):
    """Remove consecutive duplicate points from every section in the tree."""
    for r in list(roots):
        for section in r.wholetree:
            
            # some is allowed to have a single point
            if len(section.points) < 2 and section.section_type != "soma":
                delete = True

                # the simplest solution
                # check the number of children
                match len(section.children):
                    case 0:
                        section.disconnect()
                        delete = False
                    case 1:
                        if section.parent:
                            section.children[0].connect(section.parent, relation="parent")
                            section.disconnect()
                            delete = False

                if delete:
                    # check the last point of the parent
                    if section.parent:
                        # do not consider the first point
                        for i in range(1, len(section.parent.points)):
                            first_point = section.parent.points[-i]
                            if (section.points[0] != first_point).any():
                                section.points.insert(0, first_point)
                                section.parent.points = section.parent.points[:-i+1]
                                
                                delete = False
                                
                                print(f'repair_single_point_section: added {i} point from parent'
                                      f'\tparent={len(section.parent.points)}, section={len(section.points)}')
                                break
                        
                    if section.children:
                        # check the children points
                        min_point_length = min(len(ch.points)-1 for ch in section.children)
                        for i in range(min_point_length):
                            
                            last_point = np.mean([ch.points[i] for ch in section.children], axis=0)
                            if (section.points[-1] != last_point).any():
                                section.points.append(last_point)

                                for ch in section.children:
                                    ch.points = ch.points[i+1:]
                                    ch.points.insert(0, last_point)
                                
                                delete = False
                                
                                print(f'repair_single_point_section: added {i} point from children'
                                      f'\tchildren={[len(ch.points) for ch in section.children]}, section={len(section.points)}')

                                break


                if delete and len(section.children):
                    raise Exception(f'We cannot delete a section with a single point {len(section.points)} which have children')


            
def delete_consecutive_duplicate_points(roots):
    """Remove consecutive duplicate points from every section in the tree."""
    for r in roots.copy():
        for section in r.wholetree:
            if len(section.points) < 2:
                continue

            points = [section.points[0]]

            for point in section.points[1:]:
                if (point != points[-1]).any():
                    points.append(point)

            section.points = points

def delete_sections(roots, forbidden_section_types):            
    for r in roots.copy():
        for section in r.wholetree.copy():

            # if a section is not of interested it is disconnected
            if section.section_type in forbidden_section_types:
                section.disconnect()
                
                if section in roots:
                    roots.remove(section)
                    
                #print(section.section_type, "deleted")

                continue

            # if it does not have parent it is a root
            if not section.parent and section not in roots:
                roots.append(section)
                #print(section.section_type, "without parent appended as root")
                

def translate_sections(sections):        
    for section in sections.copy():
        if section.parent is None:
            target = np.zeros(3)
        else:
            target = section.parent.points[-1]
            
        section.points = misc.translate_points(section.points, section.points[0], target=target)
        
        translate_sections(section.children)

def process_soma(roots):
    for r in roots.copy():
        for section in r.wholetree:
            if section.section_type == "soma":
                section.points = [np.zeros(3)]

def merge_single_children(roots):
    for section in roots.copy():
        while len(section.children) == 1 and section.section_type == section.children[0].section_type:
            # merge point
            section.points += section.children[0].points[1:]

            # copy children
            cont_section = section.children[0]

            # disconnect section
            section.disconnect_from_children()

            
            for ch in cont_section.children.copy():
                ch.disconnect_from_parent()
                ch.connect(section, relation="parent")
            
        # visit children
        merge_single_children(section.children)


def merge_somata(roots):
    first_soma = None
    for r in roots.copy():
        for section in r.wholetree.copy():
            if section.section_type == "soma":
                
                if section.parent and section.parent.section_type != "soma":
                    raise Exception("soma has non soma as parent")

                if not first_soma:
                    first_soma = section
                    continue

                for ch in section.children:
                    ch.disconnect_from_parent()
                    ch.connect(first_soma, relation="parent")

                if section in roots:
                    roots.remove(section)
                continue

    if not first_soma:
        print('No soma was found!')
        return
    
    for r in roots.copy():
        for section in r.wholetree.copy():
            if section.section_type != "soma" and not section.parent:
                section.connect(first_soma, relation="parent")

                if section in roots:
                    roots.remove(section)
                print(section.section_type, "connected to a soma")
                

def process_morphology(roots, delete_section_types=None):
    """Delete selected section types and optionally merge same-type single-child sections."""
    # 1. delete section type that are not of interest
    delete_sections(roots, delete_section_types)

    # 2. check all the sections and delete duplicated consecutive points
    delete_consecutive_duplicate_points(roots)

    # 3. delete section with a single point
    repair_single_point_section(roots)

    # 4. replace some with point centered on the origin
    process_soma(roots)

    # 5. translate sections
    translate_sections(roots)

    # 6. merge single children
    merge_single_children(roots)

    # 7. merge all soma sections
    merge_somata(roots)
    


def _normalize_section_types(section_types):
    if type(section_types) == str:
        return [section_types]
    elif type(section_types) == list:
        for s in section_types:
            if type(s) != str:
                raise TypeError("Inappropriate section type")
    else:
        raise TypeError("Inappropriate section type: it should be a string or a list of strings")
        
    return section_types

    
def load_morphologies(directory, root_section_types=None, delete_section_types="unknown"):
    """Load and process morphologies from all SWC files in a directory."""
    
    files = sorted(Path(directory).glob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

    if root_section_types:
        root_section_types = _normalize_section_types(root_section_types)
        
    if delete_section_types:
        delete_section_types = _normalize_section_types(delete_section_types)

        
    morphologies = []

    for filename in files:
        print(filename)
        
        # neuron is represented as a list of roots
        m = read_swc(filename)

        # preprocess morphology
        process_morphology(m, delete_section_types)

        # append morphology
        morphologies.append({
            'filename':filename,
            'morphology':m
            })

    return morphologies
