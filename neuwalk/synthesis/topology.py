import numpy as np

from ..sampling import EventSampler
from ..profiles import NeuriteProfile
from ._progressive_sholl_synthesis import synthesize_progressive

from .. import misc

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
        bifurcation_count=None,
        primary_count_range=None,
        no_bifurcation_bins=None,
        no_annihilation_bins=None,
        with_soma=True,
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
        bifurcation_count : dict, optional
            Bifurcation-count statistics used by the main sampler.
        primary_count_range : dict, optional
            Minimum and maximum numbers of primary neurites.
        no_bifurcation_bins : array-like, optional
            Main bins where bifurcation is disabled.
        no_annihilation_bins : array-like, optional
            Main bins where annihilation is disabled.
        with_soma : bool, default True
            If ``True``, ``initialize`` creates a soma section and connects
            every primary root to it as a child; the soma is then exposed
            through the ``soma`` property and ``roots`` becomes
            unavailable. If ``False`` (default), the primary roots are
            created as independent sections with no parent; they are
            exposed through the ``roots`` property and ``soma`` becomes
            unavailable.
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

        self.sholl_plot_constraint = sholl_plot
        self.bifurcation_count_constraint = bifurcation_count
        self.primary_count_range_constraint = primary_count_range

        self.main_event_sampler = EventSampler(
            rng=rng,
            step_size=step_size,
            bin_size=bin_size,
            sholl_plot=sholl_plot,
            bifurcation_density=bifurcation_density,
            annihilation_density=annihilation_density,
            bifurcation_count=bifurcation_count,
            primary_count_range=primary_count_range,
            no_bifurcation_bins=no_bifurcation_bins,
            no_annihilation_bins=no_annihilation_bins,
        )
        self.event_sampler = self.main_event_sampler

        self.with_soma = bool(with_soma)
        self._roots = []
        self._soma = None
        self.initialized = False
        self.synthesis_logs = []

    @property
    def roots(self):
        """
        Return the independent primary root sections.

        Raises
        ------
        RuntimeError
            If ``with_soma`` is True. In that case the primary roots are
            attached to a soma; use ``soma`` instead.
        """
        if self.with_soma:
            raise RuntimeError(
                "with_soma is enabled: the primary roots are attached to "
                "a soma. Use `soma` instead of `roots`."
            )

        return self._roots

    @property
    def soma(self):
        """
        Return the soma section connecting the primary roots.

        Raises
        ------
        RuntimeError
            If ``with_soma`` is False. In that case no soma was created;
            use ``roots`` instead.
        """
        if not self.with_soma:
            raise RuntimeError(
                "with_soma is disabled: no soma was created. Use `roots` "
                "instead of `soma`."
            )

        return self._soma

    def initialize(self):
        """
        Create the primary neurites.

        The primary roots are always created first as independent
        sections. If ``with_soma`` was set at construction, a soma section
        is then created and every root is connected to it as a child.

        Returns
        -------
        NeuriteProfile or list of NeuriteProfile
            The soma if ``with_soma`` is True, otherwise the list of
            independent primary roots.
        """
        if self.initialized:
            raise RuntimeError(
                "The neurite tree is already initialized."
            )

        primary_count = self.main_event_sampler.sample_primary_neurite_count()

        self._roots = [
            NeuriteProfile(
                step_size=self.step_size,
                section_type=self.section_type,
            )
            for _ in range(primary_count)
        ]

        if self.with_soma:
            self._soma = NeuriteProfile(
                step_size=self.step_size,
                section_type="soma",
            )

            for root in self._roots:
                root.connect(self._soma, relation="parent")

        self.initialized = True

        return self.soma if self.with_soma else self.roots
    
    def _iter_sections(self):
        """Iterate over every section in every primary root."""
        for root in self._roots:
            yield from root.subtree

    def sholl_plot(self, max_distance=None):
        """Return the sum of the Sholl plots over all primary roots."""
        if not self._roots:
            return np.zeros(1, dtype=int)

        plots = [
            np.asarray(
                root.sholl_plot(
                    bin_size=self.bin_size,
                    max_distance=max_distance,
                ),
                dtype=int,
            )
            for root in self._roots
        ]

        size = max(len(plot) for plot in plots)
        summed_plot = np.zeros(size, dtype=int)

        for plot in plots:
            summed_plot[:len(plot)] += plot

        return summed_plot

    def bifurcation_count(self):
        """Return the total bifurcation count over all primary roots."""
        return sum(
            root.bifurcation_count
            for root in self._roots
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
        NeuriteProfile or list of NeuriteProfile
            The soma if ``with_soma`` is True, otherwise the list of
            synthesized primary roots.
        """
        return synthesize_progressive(
            tree=self,
            n_std=n_std,
            max_attempts_per_window=max_attempts_per_window,
            max_total_attempts=max_total_attempts,
            verbose=verbose,
        )

    def synthesize(self, max_steps=None, distance_limit=None):
        """
        Synthesize the tree and store the sampled events.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of synthesis sweeps. If ``None``, synthesis
            continues until no active neurites remain. Mutually exclusive
            with ``distance_limit``.
        distance_limit : float, optional
            Maximum path distance from the root that a neurite may reach.
            A neurite stops being advanced once its tip distance
            (``distance_from_root + length``) exceeds this value.
            Synthesis stops once every active neurite has exceeded it.
            Mutually exclusive with ``max_steps``.

        Returns
        -------
        NeuriteProfile or list of NeuriteProfile
            The soma if ``with_soma`` is True, otherwise the list of
            synthesized primary roots.
        """
        if max_steps is not None and distance_limit is not None:
            raise ValueError(
                "Provide either max_steps or distance_limit, not both."
            )

        if max_steps is not None:
            if not isinstance(max_steps, int):
                raise TypeError(
                    "max_steps must be an integer or None."
                )

            if max_steps < 0:
                raise ValueError(
                    "max_steps cannot be negative."
                )

        if distance_limit is not None:
            if (
                not isinstance(distance_limit, (int, float))
                or isinstance(distance_limit, bool)
            ):
                raise TypeError(
                    "distance_limit must be a number or None."
                )

            if distance_limit < 0:
                raise ValueError(
                    "distance_limit cannot be negative."
                )

        synthesis_log = []

        if not self.initialized:
            self.initialize()
            active_neurites = list(self._roots)

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

                if (
                    distance_limit is not None
                    and neurite.distance_from_root + neurite.length
                        > distance_limit
                ):
                    continue

                
                event = self.event_sampler.sample_event(neurite)

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

                elif event == "annihilate":
                    neurite.annihilate()

                else:
                    raise RuntimeError(
                        f"Unknown synthesis event: {event!r}."
                    )

            active_neurites = misc.permute(self.rng, next_active_neurites)
            step += 1

        self.synthesis_logs.append(synthesis_log)

        return self.soma if self.with_soma else self.roots

        

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

            elif event == "annihilate":
                neurite.undo_annihilate()

            elif event == "initialize":
                self._roots = []
                self._soma = None
                self.initialized = False

            else:
                raise RuntimeError(
                    f"Unknown synthesis event: {event!r}."
                )

    def describe(self):
        """Print synthesized and experimental topology statistics."""
        synthesized_primary_count = len(self._roots)

        if self.primary_count_range_constraint is None:
            experimental_primary_count = "N/A"
        else:
            experimental_primary_count = (
                f"{self.primary_count_range_constraint['min']:.1f}–"
                f"{self.primary_count_range_constraint['max']:.1f}"
            )

        print(
            "Initial primary dendrites:\t"
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
                f"{self.bifurcation_count_constraint['mean']:.1f} ± "
                f"{self.bifurcation_count_constraint['std']:.1f}"
            )

        print(
            "Bifurcation count:\t\t"
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

        print("\nSholl plot:")
        print(
            "radius\tsynthesized\t"
            "experimental mean\texperimental std"
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
                f"\t{synthesized:.1f}  "
                f"\t\t{mean:.1f}  "
                f"\t\t\t{std:.1f}"
            )
