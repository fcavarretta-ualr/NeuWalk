from ..sampling import EventSampler
from ..profiles import NeuriteProfile
from ._progressive_sholl_synthesis import synthesize_progressive

class TopologySynthesizer:
    """Represent and synthesize a neurite tree profile."""

    def __init__(
        self,
        rng,
        step_size,
        bin_size,
        section_type=None,
        sholl_plot=None,
        bifurcation_density=None,
        annihilation_density=None,
        bifurcation_internal_density=None,
        bifurcation_count=None,
        primary_count_range=None,
        no_bifurcation_bins=None,
        no_annihilation_bins=None,
        internal_event_sampler_parameters=None,
    ):
        """
        Initialize the tree profile and its event samplers.

        Parameters
        ----------
        rng : numpy.random.Generator-like
            Random number generator.
        step_size : float
            Length represented by one synthesis step.
        bin_size : float
            Width of each radial bin.
        section_type : str, optional
            Section type assigned to primary neurites.
        sholl_plot : dict, optional
            Sholl statistics used by the main event sampler.
        bifurcation_density : array-like, optional
            Main radial bifurcation density.
        annihilation_density : array-like, optional
            Main radial annihilation density.
        bifurcation_internal_density : float or array-like, optional
            Main radial internal-bifurcation density.
        bifurcation_count : dict, optional
            Bifurcation-count statistics used by the main sampler.
        primary_count_range : dict, optional
            Minimum and maximum numbers of primary neurites.
        no_bifurcation_bins : array-like, optional
            Main bins where bifurcation is disabled.
        no_annihilation_bins : array-like, optional
            Main bins where annihilation is disabled.
        internal_event_sampler_parameters : dict, optional
            Parameters used to initialize the event sampler for internal
            branches. ``rng``, ``step_size``, and ``bin_size`` default to the
            values used by the main event sampler.
        """
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        self.rng = rng
        self.step_size = float(step_size)
        self.bin_size = float(bin_size)
        self.section_type = section_type

        self.sholl_plot_constraint = sholl_plot

        self.main_event_sampler = EventSampler(
            rng=rng,
            step_size=step_size,
            bin_size=bin_size,
            sholl_plot=sholl_plot,
            bifurcation_density=bifurcation_density,
            annihilation_density=annihilation_density,
            bifurcation_internal_density=bifurcation_internal_density,
            bifurcation_count=bifurcation_count,
            primary_count_range=primary_count_range,
            no_bifurcation_bins=no_bifurcation_bins,
            no_annihilation_bins=no_annihilation_bins,
        )

        self.event_sampler = self.main_event_sampler
        self.internal_event_sampler = None

        if internal_event_sampler_parameters is not None:
            if not isinstance(
                internal_event_sampler_parameters,
                dict,
            ):
                raise TypeError(
                    "internal_event_sampler_parameters must be "
                    "a dictionary."
                )

            parameters = dict(
                internal_event_sampler_parameters
            )

            parameters.setdefault("rng", rng)
            parameters.setdefault("step_size", step_size)
            parameters.setdefault("bin_size", bin_size)

            self.internal_event_sampler = EventSampler(
                **parameters
            )

        self.soma = NeuriteProfile(
            step_size=step_size,
            section_type="soma",
        )

        self.initialized = False
        self.synthesis_logs = []

    def initialize(self):
        """
        Create the primary neurites.

        Returns
        -------
        list of NeuriteProfile
            Created primary neurites.
        """
        if self.initialized:
            raise RuntimeError(
                "The neurite tree is already initialized."
            )

        primary_count = (
            self.main_event_sampler.sample_primary_neurite_count()
        )

        primary_neurites = self.soma.create_primary_dendrites(
            number=primary_count,
            section_type=self.section_type,
        )

        self.initialized = True

        return primary_neurites
    
    def synthesize_progressive(
        self,
        n_std=1.0,
        max_attempts_per_window=1,
        max_total_attempts=1000,
        verbose=False,
    ):
        """
        Synthesize the tree using progressive Sholl-bin backtracking.

        Parameters
        ----------
        n_std : float, default 1
            Number of standard deviations allowed around each Sholl mean.
        max_attempts_per_window : int, default 1
            Number of attempts before expanding the rollback window.
        max_total_attempts : int, default 1000
            Maximum number of total regeneration attempts.
        verbose : bool, default False
            Print synthesis and rollback progress.

        Returns
        -------
        NeuriteProfile
            Soma containing the synthesized tree.
        """
        return synthesize_progressive(
            tree=self,
            n_std=n_std,
            max_attempts_per_window=max_attempts_per_window,
            max_total_attempts=max_total_attempts,
            verbose=verbose,
        )

    def synthesize(self, max_steps=None):
        """
        Synthesize the tree and store the sampled events.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of synthesis sweeps. If ``None``, synthesis
            continues until no active neurites remain.

        Returns
        -------
        NeuriteProfile
            Soma containing the synthesized tree.
        """
        if max_steps is not None:
            if not isinstance(max_steps, int):
                raise TypeError(
                    "max_steps must be an integer or None."
                )

            if max_steps < 0:
                raise ValueError(
                    "max_steps cannot be negative."
                )

        synthesis_log = []

        if not self.initialized:
            active_neurites = self.initialize()

            self.synthesis_logs.append(
                [
                    {
                        "neurite": self.soma,
                        "event": "initialize",
                    }
                ]
            )
        else:
            active_neurites = [
                neurite
                for neurite in self.soma._iter_sections()
                if (
                    neurite.active
                    and neurite.section_type != "soma"
                )
            ]

        step = 0

        while active_neurites:
            if (
                max_steps is not None
                and step >= max_steps
            ):
                break

            next_active_neurites = []

            for neurite in active_neurites:
                if not neurite.active:
                    continue

                event = self.event_sampler.sample_event(
                    neurite
                )

                synthesis_log.append(
                    {
                        "neurite": neurite,
                        "event": event,
                    }
                )

                if event == "elongate":
                    neurite.elongate()
                    next_active_neurites.append(neurite)

                elif event == "bifurcate":
                    children = neurite.bifurcate()
                    next_active_neurites.extend(children)

                elif event == "bifurcate_internal":
                    children = (
                        neurite.bifurcate_internal()
                    )

                    # The first child continues synthesis. The second child
                    # remains inactive until activate_internal_branches().
                    next_active_neurites.append(
                        children[0]
                    )

                elif event == "annihilate":
                    neurite.annihilate()

                else:
                    raise RuntimeError(
                        f"Unknown synthesis event: {event!r}."
                    )

            active_neurites = next_active_neurites
            step += 1

        self.synthesis_logs.append(synthesis_log)

        return self.soma

    
    def _neurite_is_secondary(self, neurite):
        """ check whether the neurite is seconday by comparing bifurcation and annihilation rates """
        # we want to perform the check only on neurites which are annihilated
        assert not neurite.active and not neurite.has_children()

        # check the probability
        pb, pa, pbi = self.event_sampler._probability_fn(neurite)
        
        pe = 1 - pb - pa # calculating elongation probability whici also accounts for pbi

        # if annihilation occur where annihilation is dominant
        # the neurite is not secondary
        return not ( pa >= pb and pa >= pe )
        
    def activate_internal_branches(self):
        """
        Activate inactive branches created by internal bifurcations.

        The internal event sampler becomes active when at least one internal
        branch is activated. The activation and sampler change are recorded
        as one reversible log entry.

        Returns
        -------
        list of NeuriteProfile
            Internal branches that were activated.
        """
        if self.internal_event_sampler is None:
            raise RuntimeError(
                "No internal event sampler was configured."
            )

        activation_log = []
        activated_branches = []

        previous_event_sampler = self.event_sampler

        for neurite in self.soma._iter_sections():
            if not neurite.internal_bifurcation:
                continue

            if len(neurite.children) != 2:
                raise RuntimeError(
                    "An internal bifurcation must have "
                    "exactly two children."
                )

            branch = neurite.children[1]

            if not branch.active:
                branch.active = True
                activated_branches.append(branch)

                activation_log.append(
                    {
                        "neurite": branch,
                        "event": "activate_internal_branch",
                    }
                )

        if activation_log:
            self.event_sampler = (
                self.internal_event_sampler
            )

            activation_log.append(
                {
                    "neurite": self.soma,
                    "event": "switch_event_sampler",
                    "previous_event_sampler": (
                        previous_event_sampler
                    ),
                }
            )

            self.synthesis_logs.append(
                activation_log
            )

        return activated_branches

    def undo_synthesize(self):
        """
        Undo the most recent synthesis or activation log.

        Events are undone in reverse order. Initialization removes the primary
        neurites. Internal-branch activation is reversed by deactivating the
        branches and restoring the previously active event sampler.
        """
        if not self.synthesis_logs:
            raise RuntimeError(
                "No synthesis is available to undo."
            )

        synthesis_log = self.synthesis_logs.pop()

        for record in reversed(synthesis_log):
            neurite = record["neurite"]
            event = record["event"]

            if event == "elongate":
                neurite.undo_elongate()

            elif event == "bifurcate":
                neurite.undo_bifurcate()

            elif event == "bifurcate_internal":
                neurite.undo_bifurcate_internal()

            elif event == "annihilate":
                neurite.undo_annihilate()

            elif event == "switch_event_sampler":
                if neurite is not self.soma:
                    raise RuntimeError(
                        "The sampler-switch record must "
                        "reference the soma."
                    )

                self.event_sampler = record[
                    "previous_event_sampler"
                ]

            elif event == "activate_internal_branch":
                if not neurite.active:
                    raise RuntimeError(
                        "The internal branch is already inactive."
                    )

                if neurite.children:
                    raise RuntimeError(
                        "Cannot deactivate an internal branch "
                        "while it has synthesized children. Undo "
                        "its subsequent synthesis first."
                    )

                neurite.active = False

            elif event == "initialize":
                if neurite is not self.soma:
                    raise RuntimeError(
                        "The initialization record must "
                        "reference the soma."
                    )

                self.soma.children = []
                self.initialized = False

            else:
                raise RuntimeError(
                    f"Unknown synthesis event: {event!r}."
                )

    def use_main_event_sampler(self):
        """Set the main event sampler as active."""
        self.event_sampler = self.main_event_sampler

    def use_internal_event_sampler(self):
        """Set the internal-branch event sampler as active."""
        if self.internal_event_sampler is None:
            raise RuntimeError(
                "No internal event sampler was configured."
            )

        self.event_sampler = self.internal_event_sampler
