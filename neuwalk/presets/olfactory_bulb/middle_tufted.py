from . import _generation

def generate(seed, **kwargs):
    return _generation.generate(seed, "middle_tufted", **kwargs)
