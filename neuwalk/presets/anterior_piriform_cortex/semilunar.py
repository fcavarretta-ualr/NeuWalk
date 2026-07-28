from . import _generation

def generate(seed, **kwargs):
    return _generation.generate(seed, "semilunar", **kwargs)
