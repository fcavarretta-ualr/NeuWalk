from pathlib import Path

import numpy as np

from .. import misc
from ..io import read_swc
from ..core.section import Section, Neuron

def _all_sections(roots, non_soma_only=True):
    """ iterate all the sections """
    ret = []
    for root in roots.copy():
        ret += [section for section in root.subtree if not non_soma_only or section.label != "soma"]
    return ret

    
def repair_sections(roots, tolerance=0, verbose=True):
    """Remove consecutive duplicate points from every section in the tree."""

    # Fix all leaves
    for section in _all_sections(roots):
        if section.label != "soma" and not section.has_children and section.length <= tolerance:
            section.disconnect()
               
                    
    # merge single children sections
    all_sections = _all_sections(roots)
    
    while len(all_sections):
        # get the first node
        section = all_sections.pop(0)

        # merge as long as we have one child
        while len(section.children) == 1:
            child = section.children[0]

            # disconnect and remove from the list
            child.disconnect_from_parent()
            all_sections.remove(child)

            # connect points
            section.points += child.points[1:]

            # re-arrange connectivity
            for ch in child.children:
                ch.disconnect_from_parent()
                ch.connect(section, relation="parent")


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
    for section in _all_sections(roots):
        if len(section.points) < 2:
            continue

        points = [section.points[0]]

        for point in section.points[1:]:
            if (point != points[-1]).any():
                points.append(point)

        section.points = points



def delete_sections(roots, forbidden_labels):

    # disconnect all the sections which not selected by type
    # which equivalent to delete them as none reference to them
    
    for section in _all_sections(roots):

        # if a section is not of interested it is disconnected
        if section.label in forbidden_labels:
            section.disconnect()

            # if it is a root, it should be delete from the list
            if section in roots:
                roots.remove(section)

            continue

        # if a section has no parent should be in the roots
        if not section.parent and section not in roots:
            roots.append(section)

                

def translate_sections(roots):
    for root in roots:
        source = root.points[0].copy()
        for section in root.subtree:
            section.points = [p-source for p in section.points]

    

                

def process_soma(roots):
    soma = Section(label="soma")

    # check soma integrity
    for section in _all_sections(roots, non_soma_only=False):              
        # merge all the somata
        if section.label == "soma":
            if section.parent and section.parent.label != "soma":
                raise ValueError("Soma has a non-soma parent")

            soma.points += section.points

    # calculate baricenter of the soma
    # so we have somata made by a single point
    soma.points = [np.mean(soma.points, axis=0)]

    # check soma integrity
    for section in _all_sections(roots, non_soma_only=True):              
        # delete somata
            # disconnect from previous somata
            if section.parent and section.parent.label == "soma":
                section.disconnect_from_parent()

            # if it has not parent connect with the new soma
            if not section.parent:
                section.connect(soma, relation="parent")
                section.points.insert(0, soma.points[0])
                
    # revise root list
    roots.clear()
    roots.append(soma)

                

def process_morphology(roots, delete_labels=None):
    """Delete selected labels and optionally merge same-label single-child sections."""
    # 1. delete labels that are not of interest
    # listed in delete_labels
    # orphan sections not in delete labels
    # are returned as roots
    delete_sections(roots, delete_labels)

    # 2. check all the sections and delete duplicated consecutive points
    delete_consecutive_duplicate_points(roots)

    #if "soma" not in delete_labels:
    # 3. replace some with point centered on the origin
    process_soma(roots)
    
    # 3. delete section with a single point
    # we are assuming that all sections have their connectivity already fixed
    # and all duplicated points removed
    #repair_sections(roots)
    
    # 5. translate sections
    translate_sections(roots)

    


    


def _normalize_labels(labels):
    if type(labels) == str:
        return [labels]
    elif type(labels) == list:
        for s in labels:
            if type(s) != str:
                raise TypeError("Inappropriate label")
    else:
        raise TypeError("Inappropriate label: it should be a string or a list of strings")
        
    return labels

    
def load_morphologies(directory, delete_labels="unknown", return_file_names=False):
    """Load and process morphologies from all SWC files in a directory."""
    
    files = sorted(Path(directory).rglob("*.swc"))

    if not files:
        raise ValueError(f"No SWC files found in {directory}.")

        
    morphologies = []

    for filename in files:
        
        # neuron is represented as a list of roots
        m = read_swc(filename)
        # preprocess morphology
        process_morphology(m, delete_labels)
        
        # append morphology
        if m:
            m = Neuron(m)
            morphologies.append((filename, m) if return_file_names else m)

    return morphologies
