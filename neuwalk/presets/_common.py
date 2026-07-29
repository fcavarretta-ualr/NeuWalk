from .. import misc
from ..synthesis import TopologySynthesizer, MorphologySynthesizer


def synthesize_topologies(
    all_params,
    seed,
    step_size,
    n_std,
    max_attempts_per_window,
    max_total_attempts,
    verbose,
    with_soma=None,
):
    """
    Synthesize the branching-and-annihilating topology for each section type.

    This is the loop duplicated across the neocortex, olfactory_bulb, and
    anterior_piriform_cortex generation code.

    Parameters
    ----------
    all_params : dict
        Mapping of section type to its ``TopologySynthesizer`` parameters.
    with_soma : None, bool, or callable, default None
        If ``None`` (default), ``with_soma`` is not passed to
        ``TopologySynthesizer`` at all, so its own default applies. If a
        bool, applies uniformly to every section type. If a callable
        ``section_type -> bool``, resolved separately for each section
        type (e.g. every section type except one kind).

    Returns
    -------
    dict
        Mapping of section type to ``{"topology": TopologySynthesizer}``.
    """
    ret = {}

    for section_type, params in all_params.items():
        if verbose:
            print(f"Elaboration of {section_type}")
            print(f"\tGenerating Branching-and-annihilating profile...", end="")

        extra_kwargs = {}
        if with_soma is not None:
            extra_kwargs["with_soma"] = (
                with_soma(section_type) if callable(with_soma) else with_soma
            )

        topol_synthesizer = TopologySynthesizer(
            misc.Random(seed),
            step_size=step_size,
            section_type=section_type,
            **extra_kwargs,
            **params
        )

        topol_synthesizer.synthesize_progressive(
            n_std=n_std,
            max_attempts_per_window=max_attempts_per_window,
            max_total_attempts=max_total_attempts,
            verbose=verbose
        )

        ret[section_type] = {
            'topology': topol_synthesizer,
            }

        if verbose:
            print("done\n")

    return ret


def synthesize_dendrite_tree(
    ret,
    seed,
    section_type,
    theta,
    phi,
    axis_direction,
    bifurcation_bias,
    elongation_bias,
    bifurcation_internal_bias=None,
    soma=None,
):
    """
    Build and synthesize one dendrite tree's morphology from its topology's
    soma. This is the ``MorphologySynthesizer`` pattern duplicated between
    the neocortex and anterior_piriform_cortex apical/basal dendrite trees.

    Returns
    -------
    MorphologySynthesizer
        The synthesizer, after ``synthesize`` has been called on it.
    """
    synthesizer = MorphologySynthesizer(
        root=ret[section_type]['topology'].soma,
        rng=misc.Random(seed),
        theta=theta,
        phi=phi,
        axis_direction=axis_direction,
        bifurcation_bias=bifurcation_bias,
        bifurcation_internal_bias=bifurcation_internal_bias,
        elongation_bias=elongation_bias,
    )

    synthesizer.synthesize(soma=soma)

    return synthesizer
