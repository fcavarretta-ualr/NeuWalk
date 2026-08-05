# neuwalk/core/topology/__init__.py

from .section_synthesizer import SectionSynthesizer
from .assign_sections import connect_internal_branches
from .merge import merge_trees

__all__ = [
    "SectionSynthesizer",
    "connect_internal_branches",
    "merge_trees",
]
