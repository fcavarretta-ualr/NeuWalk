def merge_profiles(profile_roots, step_size):
    """Attach profile roots to a new soma profile."""
    from morphgenpy.profiles import NeuriteProfile

    soma = NeuriteProfile(step_size, section_type="soma")
    for root in profile_roots:
        root.connect(soma, relation="parent")
    return soma


def merge_somas(*somas):
    """Move the roots of several synthesized somas under one soma."""
    if not somas:
        raise ValueError("At least one soma is required.")

    soma = somas[0].clone()
    soma.children = []

    for source in somas:
        for root in list(source.children):
            root.disconnect_from_parent()
            root.connect(soma, relation="parent")

    return soma


def remove_soma(soma, container=list):
    """Detach the primary neurites and place them in ``container``."""
    roots = list(soma.children)
    for root in roots:
        root.disconnect_from_parent()
    return container(roots)
