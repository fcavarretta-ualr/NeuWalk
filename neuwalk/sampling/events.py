import numpy as np

from .. import estimation as es


class EventSampler:
    """
    Sample section synthesis events from radial event densities.

    Initialize using either ``sholl_plot`` or both
    ``bifurcation_density`` and ``annihilation_density``.
    """

    EVENTS = (
        "bifurcate",
        "annihilate",
        "elongate",
    )

    def __init__(
        self,
        rng,
        step_size,
        bin_size,
        sholl_plot=None,
        bifurcation_density=None,
        annihilation_density=None,
        bifurcation_count=None,
        primary_count_range=None,
        no_bifurcation_bins=None,
        no_annihilation_bins=None,
    ):
        """
        Initialize the event sampler.

        Parameters
        ----------
        rng : numpy.random.Generator-like
            Random number generator providing ``random()``.
        step_size : float
            Length represented by one synthesis step.
        bin_size : float
            Width of each radial bin.
        sholl_plot : dict, optional
            Sholl statistics used to estimate event densities.
        bifurcation_density : array-like, optional
            Radial bifurcation density.
        annihilation_density : array-like, optional
            Radial annihilation density.
        bifurcation_count : dict, optional
            Bifurcation-count statistics used with ``sholl_plot``.
        primary_count_range : dict, optional
            Minimum and maximum numbers of primary sections.
        no_bifurcation_bins : array-like, optional
            Bins where bifurcation is disabled.
        no_annihilation_bins : array-like, optional
            Bins where annihilation is disabled.
        """
        if not hasattr(rng, "random"):
            raise TypeError("rng must provide a random() method.")

        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        if step_size > bin_size:
            raise ValueError(
                "step_size cannot exceed bin_size "
                f"({step_size} > {bin_size})."
            )

        self.rng = rng
        self.step_size = float(step_size)
        self.bin_size = float(bin_size)

        (
            self.bifurcation_density,
            self.annihilation_density
        ) = self._initialize_densities(
            sholl_plot=sholl_plot,
            bifurcation_density=bifurcation_density,
            annihilation_density=annihilation_density,
            bifurcation_count=bifurcation_count,
            no_bifurcation_bins=no_bifurcation_bins,
            no_annihilation_bins=no_annihilation_bins
        )



        self._validate_density_shapes()

        self.primary_count_min = None
        self.init_count_cdf = None

        if primary_count_range is not None:
            self._initialize_primary_count(
                sholl_plot,
                primary_count_range,
            )

    def _initialize_densities(
        self,
        sholl_plot,
        bifurcation_density,
        annihilation_density,
        bifurcation_count,
        no_bifurcation_bins,
        no_annihilation_bins
    ):
        """Initialize bifurcation and annihilation densities."""
        has_sholl_plot = sholl_plot is not None
        has_bifurcation_density = bifurcation_density is not None
        has_annihilation_density = annihilation_density is not None

        if has_bifurcation_density != has_annihilation_density:
            raise ValueError(
                "bifurcation_density and annihilation_density "
                "must be provided together."
            )

        has_direct_densities = (
            has_bifurcation_density
            and has_annihilation_density
        )

        if has_sholl_plot == has_direct_densities:
            raise ValueError(
                "Provide either sholl_plot or both bifurcation_density "
                "and annihilation_density, but not both."
            )

        if has_direct_densities:
            return (
                np.asarray(bifurcation_density, dtype=float),
                np.asarray(annihilation_density, dtype=float),
            )

        rates = es.event_rates(
            self.bin_size,
            sholl_plot,
            self.step_size,
            bifurcation_count=bifurcation_count,
            no_bifurcation_bins=no_bifurcation_bins,
            no_annihilation_bins=no_annihilation_bins
        )

        return (
            np.asarray(
                rates["bifurcation_rate"],
                dtype=float,
            ),
            np.asarray(
                rates["annihilation_rate"],
                dtype=float,
            )        
        )


    def _validate_density_shapes(self):
        """Check that all density arrays have the same shape."""
        expected_shape = self.bifurcation_density.shape

        if self.annihilation_density.shape != expected_shape:
            raise ValueError(
                "annihilation_density must have shape "
                f"{expected_shape}; got "
                f"{self.annihilation_density.shape}."
            )


    def _initialize_primary_count(
        self,
        sholl_plot,
        primary_count_range,
    ):
        """Initialize the primary-section count distribution."""
        if sholl_plot is None:
            raise ValueError(
                "primary_count_range requires sholl_plot."
            )

        self.primary_count_min = int(
            primary_count_range["min"]
        )

        probabilities = es.initial_count_pmf(
            sholl_plot["mean"][0],
            sholl_plot["std"][0],
            primary_count_range["min"],
            primary_count_range["max"],
        )

        self.init_count_cdf = np.cumsum(probabilities)

    def _integrate_density(
        self,
        density,
        distance_start,
        distance_end,
    ):
        """
        Calculate event probability over one synthesis step.

        The step may lie in one bin or cross one bin boundary. If it crosses a
        boundary, the density of the bin containing most of the step is used for
        the entire step.

        Raises
        ------
        ValueError
            If the interval is invalid, spans more than two bins, or selects a bin
            outside the density array.
        """
        if distance_end <= distance_start:
            raise ValueError(
                "distance_end must be greater than distance_start."
            )

        if len(density) == 0:
            raise ValueError("density cannot be empty.")

        step_length = distance_end - distance_start

        # An endpoint exactly on a boundary belongs to the preceding interval.
        end_inside = np.nextafter(distance_end, distance_start)

        start_bin = int(np.floor(distance_start / self.bin_size))
        end_bin = int(np.floor(end_inside / self.bin_size))
        
        if end_bin - start_bin > 1:
            raise ValueError(
                "A synthesis step cannot span more than two bins. "
                f"Interval [{distance_start}, {distance_end}] spans "
                f"bins {start_bin} through {end_bin}."
            )

        if start_bin == end_bin:
            selected_bin = start_bin

        else:
            boundary = (start_bin + 1) * self.bin_size

            length_in_start_bin = boundary - distance_start
            length_in_end_bin = distance_end - boundary

            # Use the starting bin in case of an exact tie.
            selected_bin = (
                start_bin
                if length_in_start_bin >= length_in_end_bin
                else end_bin
            )

        if selected_bin < 0 or selected_bin > len(density) - 1:
            raise IndexError(
                f"Selected bin {selected_bin} is outside the density "
                f"range [0, {len(density) - 1}]."
            )

        return float(density[selected_bin] * step_length)

    def _probability_fn(self, section):
        """Calculate event probabilities for the latest synthesis step."""                
        distance_end = section.distance_from_root + section.length
        distance_start = distance_end - section.step_size

        if distance_end <= distance_start:
            return 0.0, 0.0, 0.0

        return (
            self._integrate_density(
                self.bifurcation_density,
                distance_start,
                distance_end,
            ),
            self._integrate_density(
                self.annihilation_density,
                distance_start,
                distance_end,
            )
        )

    def sample_event(self, section):
        """Sample one synthesis event for a section."""
        
        if not hasattr(section, "label"):
            raise TypeError(
                "section must provide a label attribute."
            )

        if section.label == "soma":
            raise RuntimeError(
                "Cannot sample a synthesis event for the soma."
            )
        
        probabilities = np.asarray(
            self._probability_fn(section),
            dtype=float,
        )

        if (
            np.isposinf(probabilities[1])
            and np.isfinite(probabilities[0]).all()
        ):
            return "annihilate"

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Invalid event probabilities: {probabilities}."
            )

        if np.any(
            (probabilities < 0.0)
            | (probabilities > 1.0)
        ):
            raise ValueError(
                "Event probabilities must be between 0 and 1: "
                f"{probabilities}."
            )

        total = probabilities.sum()

        if total > 1.0 and not np.isclose(total, 1.0):
            raise ValueError(
                f"Event probabilities sum to more than 1: {total}."
            )

        cdf = np.minimum(
            np.cumsum(probabilities),
            1.0,
        )
        X = self.rng.random()
        event_index = np.searchsorted(
            cdf,
            X,
            side="right",
        )

        return self.EVENTS[event_index]

    def sample_primary_section_count(self):
        """Sample the number of primary sections."""
        if self.init_count_cdf is None:
            raise RuntimeError(
                "The primary-count distribution is unavailable."
            )

        return int(np.searchsorted(
            self.init_count_cdf,
            self.rng.random(),
            side="right",
        ))

