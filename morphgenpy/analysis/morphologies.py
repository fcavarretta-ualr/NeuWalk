from pathlib import Path

import numpy as np

from .. import misc
from ..io import read_swc
from ..core.neurite import Neurite

def _all_sections(roots, non_soma_only=True):
    """ iterate all the sections """
    ret = []
    for root in roots.copy():
        ret += [section for section in root.subtree if not non_soma_only or section.section_type != "soma"]
    return ret

    
def repair_sections(roots, tolerance=15, verbose=True):
    """Remove consecutive duplicate points from every section in the tree."""

    # Fix all leaves
    for neurite in _all_sections(roots):
        if neurite.section_type != "soma" and not neurite.has_children and neurite.length <= tolerance:
            neurite.disconnect()
               
                    
    # merge single children sections
    all_sections = _all_sections(roots)
    
    while len(all_sections):
        # get the first node
        neurite = all_sections.pop(0)

        # merge as long as we have one child
        while len(neurite.children) == 1:
            child = neurite.children[0]

            # disconnect and remove from the list
            child.disconnect_from_parent()
            all_sections.remove(child)

            # connect points
            neurite.points += child.points[1:]

            # re-arrange connectivity
            for ch in child.children:
                ch.disconnect_from_parent()
                ch.connect(neurite, relation="parent")


    for section in _all_sections(roots):
        if len(section.points) < 2:
            success=False
            
            # interpolate with points from parent if available
            if section.parent and (section.parent.points[-1] != section.points[0]).any():
                new_point = np.mean([section.parent.points[-1], section.points[0]], axis=0)

                if np.linalg.norm(new_point - section.points[0]) <= tolerance:
                    section.points.insert(0, new_point)
                    success=True
                    print('added point from parent')

            # interpolate with points from children if available
            if section.children:
                tmp = [ch.points[0] for ch in section.children if (ch.points[0] != section.points[-1]).any()]

                if tmp:
                    new_point = (section.points[-1] + np.mean(tmp, axis=0)) * 0.5

                    if np.linalg.norm(new_point - section.points[-1]) <= tolerance:
                        section.points.append(new_point)
                        success=True
                        print('added point from children')


            if not success:
                raise Exception("There are still sections with one point.")
            

        


            
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

            #assert len(section.points) > 1, "Section has less than 2 points."


def delete_sections(roots, forbidden_section_types):
    for root in roots.copy():
        for section in root.subtree:

            # if a section is not of interested it is disconnected
            if section.section_type in forbidden_section_types:
                section.disconnect()
                
                if section in roots:
                    roots.remove(section)

                continue

            # if it does not have parent it is a root
            if not section.parent and section not in roots:
                roots.append(section)

                

def translate_sections(roots):
    for root in roots:
        source = root.points[0].copy()
        for section in root.subtree:
            section.points = [p-source for p in section.points]
            #print(section.points, source)

    

                

def process_soma(roots):
    soma = Neurite(section_type="soma")

    # check soma integrity
    for section in _all_sections(roots, non_soma_only=False):              
        # delete somata
        if section.section_type == "soma":
            if section.parent and section.parent.section_type != "soma":
                raise ValueError("Soma has a non-soma parent")


        else:
            # disconnect from previous somata
            if section.parent and section.parent.section_type == "soma":
                section.disconnect_from_parent()

            # if it has not parent connect with the new soma
            if not section.parent:
                section.connect(soma, relation="parent")
                soma.points.append(section.points[0])

    # calculate baricenter
    soma.points = [np.mean(soma.points, axis=0)]

    # revise root list
    roots.clear()
    roots.append(soma)

                

def process_morphology(roots, delete_section_types=None):
    """Delete selected section types and optionally merge same-type single-child sections."""
    # 1. delete section type that are not of interest
    delete_sections(roots, delete_section_types)

    # 2. check all the sections and delete duplicated consecutive points
    delete_consecutive_duplicate_points(roots)

    if "soma" not in delete_section_types:
        # 3. replace some with point centered on the origin
        process_soma(roots)
    
    # 3. delete section with a single point
    # we are assuming that all sections have their connectivity already fixed
    # and all duplicated points removed
    repair_sections(roots)
    
    # 5. translate sections
    translate_sections(roots)

    


    


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

    
def load_morphologies(directory, delete_section_types="unknown"):
    """Load and process morphologies from all SWC files in a directory."""
    
    files = sorted(Path(directory).glob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

        
    morphologies = []

    for filename in files:
        
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
