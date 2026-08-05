from ..random import Random
from ..core.topology import connect_internal_branches
from .topology import TopologySynthesizer
from .morphology import MorphologySynthesizer


class NeuronSynthesizer:
    """
    Orchestrate topology and morphology synthesis for a neuron made of
    several labeled section trees (e.g. ``"basal_dendrite"``,
    ``"apical_dendrite"``), including trees that are generated
    independently and then grafted onto another tree as internal
    branches (e.g. apical obliques).

    This generalizes the pattern duplicated across the neocortex,
    olfactory_bulb, and anterior_piriform_cortex preset generation code.
    """

    def __init__(
        self,
        seed,
        step_size,
        all_params,
        *,
        n_std=1.0,
        max_attempts_per_window=10,
        max_total_attempts=1000,
        verbose=False,
        obliques=None,
    ):
        """
        Synthesize the topology of every label, then graft every oblique
        onto its target's primary sections as internal branches.

        Parameters
        ----------
        seed : int
            Seed used to initialize every random-number generator created
            internally.
        step_size : float
            Length represented by one synthesis step, shared by every
            label.
        all_params : dict
            Mapping of label to its ``TopologySynthesizer`` parameters
            (``sholl_plot``, ``bifurcation_count``, ``primary_count_range``,
            ``bin_size``, ``no_bifurcation_bins``, ``no_annihilation_bins``).
        n_std, max_attempts_per_window, max_total_attempts, verbose :
            Passed to every label's ``synthesize_progressive`` call.
        obliques : dict, optional
            Mapping of oblique label to a dict describing how it attaches
            to another label's tree, with keys:

            - ``target`` : label whose primary sections the oblique's
              topology is grafted onto as internal branches.
            - ``density`` : radial density of internal branch points,
              passed to ``connect_internal_branches``.
            - ``bin_size`` : bin size for the same call. Defaults to
              ``step_size`` when omitted.

            Every oblique label is synthesized as an independent
            (soma-less) topology instead of getting its own soma.
        """
        self.seed = seed
        self.step_size = step_size
        self.obliques = dict(obliques) if obliques else {}

        for label, spec in self.obliques.items():
            if label not in all_params:
                raise ValueError(f"No parameters were given for oblique label {label!r}.")

            if spec.get("target") not in all_params:
                raise ValueError(
                    f"Oblique label {label!r} targets unknown label {spec.get('target')!r}."
                )

        def with_soma(label):
            return label not in self.obliques

        self.topologies = {}

        for label, params in all_params.items():
            if verbose:
                print(f"Elaboration of {label}")
                print("\tGenerating Branching-and-annihilating profile...", end="")

            topology_synthesizer = TopologySynthesizer(
                Random(seed),
                step_size=step_size,
                label=label,
                with_soma=with_soma(label),
                **params,
            )

            topology_synthesizer.synthesize_progressive(
                n_std=n_std,
                max_attempts_per_window=max_attempts_per_window,
                max_total_attempts=max_total_attempts,
                verbose=verbose,
            )

            self.topologies[label] = {"topology": topology_synthesizer}

            if verbose:
                print("done\n")

        self.morphologies = {}

        for label, spec in self.obliques.items():
            target = spec["target"]

            connect_internal_branches(
                self.topologies[label]["topology"].roots,
                self.topologies[target]["topology"].soma.children,
                Random(seed),
                spec["density"],
                spec.get("bin_size", step_size),
            )

    def synthesize_morphology(
        self,
        label,
        theta,
        phi,
        axis_direction,
        bifurcation_bias,
        elongation_bias,
        bifurcation_internal_bias=None,
        soma=None,
        max_steps=None,
        oblique_elongation_bias=None,
    ):
        """
        Synthesize one label's morphology tree.

        If any oblique targets ``label``, its grafted internal branches
        are synthesized right after, reusing ``elongation_bias`` unless
        ``oblique_elongation_bias`` is given.

        Parameters
        ----------
        label : str
            Label of the tree to synthesize; must be a key of
            ``all_params`` and not an oblique label.
        theta, phi, axis_direction : as in ``MorphologySynthesizer``.
        bifurcation_bias : BifurcationBias
            Bifurcation bias for this tree.
        elongation_bias : ElongationBias or sequence
            Elongation bias for this tree.
        bifurcation_internal_bias : BifurcationBias, optional
            Bifurcation bias applied at internal branch points, needed
            when an oblique targets this label.
        soma : Section, optional
            Existing soma to attach this tree to (see
            ``MorphologySynthesizer``), for sharing one soma across
            several labels.
        max_steps : int, optional
            Passed to ``synthesize``.
        oblique_elongation_bias : ElongationBias or sequence, optional
            Elongation bias used to grow any oblique grafted onto this
            label, if different from ``elongation_bias``.

        Returns
        -------
        MorphologySynthesizer
            The synthesizer, after both the main tree and (if
            applicable) its grafted obliques have been synthesized.
        """
        if label in self.obliques:
            raise ValueError(
                f"{label!r} is an oblique label; synthesize its target "
                f"({self.obliques[label]['target']!r}) instead."
            )

        synthesizer = MorphologySynthesizer(
            topology=self.topologies[label]["topology"].soma,
            rng=Random(self.seed),
            theta=theta,
            phi=phi,
            axis_direction=axis_direction,
            bifurcation_bias=bifurcation_bias,
            bifurcation_internal_bias=bifurcation_internal_bias,
            elongation_bias=elongation_bias,
            parent=soma,
        )

        synthesizer.synthesize(max_steps=max_steps)

        self.morphologies[label] = synthesizer
        self.topologies[label]["morphology"] = synthesizer

        for oblique_label, spec in self.obliques.items():
            if spec["target"] == label:
                synthesizer.synthesize(
                    max_steps=max_steps,
                    is_root_like=True,
                    elongation_bias=oblique_elongation_bias
                        if oblique_elongation_bias is not None
                        else elongation_bias,
                )

        return synthesizer
