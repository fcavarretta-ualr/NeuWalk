# neuwalk/core/topology/__init__.py

from .section_synthesizer import SectionSynthesizer
from .assign_sections import connect_internal_branches

__all__ = [
    "SectionSynthesizer",
    "connect_internal_branches"
]
