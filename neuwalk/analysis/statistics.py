from .extraction import extract_statistics
from .morphologies import load_morphologies


def load_statistics(directory, bin_size, root_section_types=None):
    """Load processed SWC morphologies and extract their statistics."""
    morphologies = load_morphologies(directory, root_section_types=root_section_types)
    return extract_statistics(morphologies, bin_size)
