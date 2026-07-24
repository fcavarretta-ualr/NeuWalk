from .apc.pyramidal import PRESET as APC_PYRAMIDAL


_PRESETS = {
    APC_PYRAMIDAL.name: APC_PYRAMIDAL,
}


def available_presets():
    """Return the names of all built-in presets."""
    return tuple(sorted(_PRESETS))


def get_preset(name):
    """Return a named preset."""
    try:
        return _PRESETS[name]
    except KeyError:
        available = ", ".join(available_presets())
        raise KeyError(
            f"Unknown preset {name!r}. Available presets: {available}"
        ) from None


def generate(name, **kwargs):
    """Generate one morphology from a named preset."""
    return get_preset(name).generate(**kwargs)


def generate_neurites(name, **kwargs):
    """Generate detached primary neurites without a soma."""
    kwargs["with_soma"] = False
    return generate(name, **kwargs)


def fit(name, *args, **kwargs):
    """Fit and save a named preset."""
    return get_preset(name).fit(*args, **kwargs)
