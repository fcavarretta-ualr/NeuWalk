import numpy as np

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
        internal_bifurcation_section_type=None,
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
        internal_bifurcation_section_type : str, optional
            Section type assigned to internal branches.
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

        assert section_type, "Specify section type"
        self.section_type = section_type

        if internal_bifurcation_section_type is None:
            internal_bifurcation_section_type = section_type
        self.internal_bifurcation_section_type = internal_bifurcation_section_type
        

        self.sholl_plot_constraint = sholl_plot
        self.bifurcation_count = bifurcation_count
        self.primary_count_range = primary_count_range

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

        self.roots = []
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

        self.roots = [
            NeuriteProfile(
                step_size=self.step_size,
                section_type=self.section_type,
            )
            for _ in range(primary_count)
        ]
        self.initialized = True

        return self.roots
    
    def _iter_sections(self):
        """Iterate over every section in every root."""
        for root in self.roots:
            yield from root._iter_sections()

    def sholl_plot(self, max_distance=None):
        """Return the sum of the Sholl plots over all roots."""
        if not self.roots:
            return np.zeros(1, dtype=int)

        plots = [
            np.asarray(
                root.sholl_plot(
                    bin_size=self.bin_size,
                    max_distance=max_distance,
                ),
                dtype=int,
            )
            for root in self.roots
        ]

        size = max(len(plot) for plot in plots)
        summed_plot = np.zeros(size, dtype=int)

        for plot in plots:
            summed_plot[:len(plot)] += plot

        return summed_plot

    def bifurcation_count(self):
        """Return the total bifurcation count over all roots."""
        return sum(
            root.bifurcation_count()
            for root in self.roots
        )

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
        list of NeuriteProfile
            Synthesized root profiles.
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
        list of NeuriteProfile
            Synthesized root profiles.
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
                        "event": "initialize",
                    }
                ]
            )
        else:
            active_neurites = [
                neurite
                for neurite in self._iter_sections()
                if neurite.active
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
                    children = neurite.bifurcate_internal(
                        section_type=(
                            self.internal_bifurcation_section_type
                        )
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

        return self.roots

    
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

       

        for neurite in self._iter_sections():

            if not neurite.internal_bifurcation:
                continue


            if not neurite.active:
                neurite.step_count += 1
                neurite.active = True
                activated_branches.append(neurite)

                activation_log.append(
                    {
                        "neurite": neurite,
                        "event": "activate_internal_branch",
                    }
                )

        if activation_log:
            previous_event_sampler = self.event_sampler
            self.event_sampler = self.internal_event_sampler

            activation_log.append(
                {
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
            event = record["event"]
            neurite = record.get("neurite")

            if event == "elongate":
                neurite.undo_elongate()

            elif event == "bifurcate":
                neurite.undo_bifurcate()

            elif event == "bifurcate_internal":
                neurite.undo_bifurcate_internal()

            elif event == "annihilate":
                neurite.undo_annihilate()

            elif event == "switch_event_sampler":
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
                self.roots = []
                self.initialized = False

            else:
                raise RuntimeError(
                    f"Unknown synthesis event: {event!r}."
                )

    def describe(self):
        """Print synthesized and experimental topology statistics."""
        synthesized_primary_count = len(self.roots)

        if self.primary_count_range is None:
            experimental_primary_count = "N/A"
        else:
            experimental_primary_count = (
                f"{self.primary_count_range['min']:.1f}–"
                f"{self.primary_count_range['max']:.1f}"
            )

        print(
            "Initial primary dendrites: "
            f"synthesized={synthesized_primary_count:.1f}, "
            f"experimental={experimental_primary_count}"
        )

        synthesized_bifurcation_count = (
            self.bifurcation_count()
        )

        if self.bifurcation_count is None:
            experimental_bifurcation_count = "N/A"
        else:
            experimental_bifurcation_count = (
                f"{self.bifurcation_count['mean']:.1f} ± "
                f"{self.bifurcation_count['std']:.1f}"
            )

        print(
            "Bifurcation count: "
            f"synthesized={synthesized_bifurcation_count:.1f}, "
            f"experimental={experimental_bifurcation_count}"
        )

        if self.sholl_plot_constraint is None:
            synthesized_sholl = self.sholl_plot()

            print("Sholl plot:")
            print("radius  synthesized")

            for index, synthesized in enumerate(
                synthesized_sholl
            ):
                print(
                    f"{index * self.bin_size:.1f}  "
                    f"{synthesized:.1f}"
                )

            return

        experimental_mean = np.asarray(
            self.sholl_plot_constraint["mean"],
            dtype=float,
        )
        experimental_std = np.asarray(
            self.sholl_plot_constraint["std"],
            dtype=float,
        )
        synthesized_sholl = self.sholl_plot(
            max_distance=(
                len(experimental_mean) - 1
            ) * self.bin_size,
        )

        print("Sholl plot:")
        print(
            "radius  synthesized  "
            "experimental mean  experimental std"
        )

        for index, (
            synthesized,
            mean,
            std,
        ) in enumerate(
            zip(
                synthesized_sholl,
                experimental_mean,
                experimental_std,
            )
        ):
            print(
                f"{index * self.bin_size:.1f}  "
                f"{synthesized:.1f}  "
                f"{mean:.1f}  "
                f"{std:.1f}"
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
