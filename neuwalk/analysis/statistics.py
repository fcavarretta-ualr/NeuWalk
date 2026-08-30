from .extraction import extract_statistics
from .morphologies import load_morphologies


def load_statistics(directory, bin_size, delete_labels="unknown"):
    """Load processed SWC morphologies and extract their statistics."""
    morphologies = load_morphologies(directory, delete_labels=delete_labels, return_file_names=False, soma_processing=False)
    return extract_statistics(morphologies, bin_size)
