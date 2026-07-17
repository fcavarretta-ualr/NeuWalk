import morphgenpy.misc as misc
import numpy as np

def translate_subtree(section, target=None):
  """ translate the points with the first one of the root coinciding with source """
  # translate the section
  section.points = misc.translate_points(section.points, section.points[0], target=target)

  # translate children and their subtrees
  for ch in section.children:
    translate_subtree(ch, target=section.points[-1])
    

def extract_neurites(sectionmorphology, section_type):
  """
  Clone a morphology and extract top-level trees of selected section types.

  A selected section becomes a retained root when it has no parent or when
  its parent's section type is different.
  """
  
  root_sections = []

  # find the roots
  for root in morphology:
    
    # let's work on a copy
    for section in root.clone().subtree:
      # found a section of interest
      if section.section_type == section_type:

        
        # if the section is of interest but parent has a different section type
        # we should disconnect and treat as an independent tree
        if section.parent and section.parent.section_type != section_type:
          section.disconnect_from_parent()

        # now if it has no parent, it is a root
        if not section.parent:          
          root_sections.append(section)

  # filter descendants from the root which are not of the same section
  for root in root_sections:
      for section in root.subtree:
        if section.section_type != section_type:
          section.disconnect_from_parent()

  # translate the root to the origin
  for root in root_sections:
      translate_subtree(section)

  return root_sections
    

    
def preprocess_morphology(morphology):
    """Split one loaded morphology into three independent cloned groups."""
    return extract_neurites(morphology, "basal_dendrites"), \
           extract_neurites(morphology, "apical_dendrites"), \
           extract_neurites(morphology, "apical_oblique")
  
if __name__ == '__main__':
  import sys

  from morphgenpy.analysis.morphologies import load_morphologies

  morphologies = load_morphologies(sys.argv[-1])

  for morphology in morphologies:
      basal_dendrites, apical_dendrites, apical_obliques = \
                       preprocess_morphology(morphology)
