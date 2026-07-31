from .extraction import extract_statistics
from .morphologies import load_morphologies


def load_statistics(directory, bin_size, root_categories=None):
    """Load processed SWC morphologies and extract their statistics."""
    morphologies = load_morphologies(directory, root_categories=root_categories)
    return extract_statistics(morphologies, bin_size)
