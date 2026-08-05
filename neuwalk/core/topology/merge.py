from .section_synthesizer import SectionSynthesizer


def _primary_sections(tree):
    """Return the primary sections of ``tree``."""
    if tree.label == "soma":
        return list(tree.children)

    return [tree]


def merge_trees(tree1, tree2):
    """
    Merge two section trees so that they share a single soma.

    Each tree may be a soma (with its own primary sections as children)
    or a single, soma-less primary section. Every primary section from
    both trees is connected, as a child, to one newly created soma;
    any soma that either tree already had is discarded (and left with
    no children) once its primary sections have been moved over.

    Parameters
    ----------
    tree1 : SectionSynthesizer
        Root of the first tree. May be a soma or a single primary
        section.
    tree2 : SectionSynthesizer
        Root of the second tree. May be a soma or a single primary
        section.

    Returns
    -------
    SectionSynthesizer
        The new soma connecting every primary section from both trees.
    """
    if not isinstance(tree1, SectionSynthesizer) or not isinstance(tree2, SectionSynthesizer):
        raise TypeError("tree1 and tree2 must be SectionSynthesizer instances.")

    if tree1 is tree2:
        raise ValueError("tree1 and tree2 must be different trees.")

    sections = _primary_sections(tree1) + _primary_sections(tree2)

    if not sections:
        raise ValueError("tree1 and tree2 must contain at least one primary section.")

    soma = SectionSynthesizer(1, label="soma")

    for section in sections:
        if section.parent is not None:
            section.disconnect_from_parent()

        section.connect(soma, relation="parent")

    return soma
