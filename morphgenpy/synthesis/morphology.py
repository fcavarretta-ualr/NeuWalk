import numpy as np

from .. import misc
from ..profiles import NeuriteProfile
from ..core.neurite import Neurite
from ..core.randomwalk import RandomWalk


class MorphologySynthesizer:
    """Generate RandomWalk trajectories from a NeuriteProfile tree."""

    def __init__(
        self,
        root,
        rng,
        origin=None,
        theta=(0.0, np.pi),
        phi=(0.0, 2.0 * np.pi),
        axis_direction=None,
        elongation_bias=None,
        bifurcation_bias=None,
        centrifugal=True,
        max_angle=np.pi / 2,
        elongation_random_weight=1.0,
        elongation_bias_weight=1.0,
    ):
        """
        Initialize the morphology synthesizer.

        Parameters
        ----------
        root : NeuriteProfile
            Root of the synthesized topology. It may be either a soma or a
            single primary neurite.
        rng : numpy.random.Generator-like
            Random number generator used by the RandomWalk objects.
        origin : array_like, optional
            Soma position. Default is ``[0, 0, 0]``.
        theta : float, sequence of two floats, or dict, optional
            Polar angle or angular range used to generate the primary
            directions. A dictionary may map a primary-dendrite count to a
            dictionary containing ``theta`` and ``phi``. The ``"default"``
            entry is used when the count is not present.
        phi : float or sequence of two floats, optional
            Azimuthal angle or angular range used to generate the primary
            directions.
        axis_direction : array_like, optional
            Direction of the local axial frame in global coordinates.
        elongation_bias : ElongationBias or sequence, optional
            Elongation bias or weighted elongation biases.
        bifurcation_bias : BifurcationBias, optional
            Bifurcation bias passed to each RandomWalk.
        centrifugal : bool, default True
            Whether RandomWalk displacement is centrifugal.
        max_angle : float, default pi / 2
            Maximum angle used when sampling elongation directions.
        elongation_random_weight : float, default 1.0
            Weight of the random elongation component.
        elongation_bias_weight : float, default 1.0
            Global weight applied to elongation biases.
        """
        if not isinstance(root, NeuriteProfile):
            raise TypeError("root must be a NeuriteProfile.")

        if origin is None:
            origin = np.zeros(3, dtype=float)

        origin = np.asarray(origin, dtype=float)

        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,).")

        if axis_direction is not None:
            axis_direction = np.asarray(axis_direction, dtype=float)

            if axis_direction.shape != (3,):
                raise ValueError("axis_direction must have shape (3,).")

            if np.isclose(np.linalg.norm(axis_direction), 0.0):
                raise ValueError("axis_direction cannot be the zero vector.")

        self.root = root
        self.rng = rng
        self.origin = origin.copy()
        self.theta = theta
        self.phi = phi
        self.axis_direction = None if axis_direction is None else axis_direction.copy()
        self.elongation_bias = elongation_bias
        self.bifurcation_bias = bifurcation_bias
        self.centrifugal = bool(centrifugal)
        self.max_angle = max_angle
        self.elongation_random_weight = elongation_random_weight
        self.elongation_bias_weight = elongation_bias_weight

    def _resolve_primary_angles(self, primary_count):
        """Resolve theta and phi independently for the primary-dendrite count."""
        theta = self.theta
        phi = self.phi

        if isinstance(theta, dict):
            theta = theta.get(
                primary_count,
                theta["default"],
            )

        if isinstance(phi, dict):
            phi = phi.get(
                primary_count,
                phi["default"],
            )

        return theta, phi

    def next_event(self, neurite):
        """
        Return the next event for a profile/walk pair.

        Parameters
        ----------
        neurite : tuple
            Pair containing ``(NeuriteProfile, RandomWalk)``.

        Returns
        -------
        str or None
            One of ``"elongate"``, ``"bifurcate"``,
            ``"bifurcate_internal"``, or ``"annihilate"``.
        """
        profile, walk = neurite
        generated_steps = len(walk.points) - 1

        if generated_steps < profile.step_count:
            return "elongate"

        if generated_steps > profile.step_count:
            raise RuntimeError(
                f"The RandomWalk ({generated_steps}) has more steps than "
                f"the corresponding NeuriteProfile ({profile.step_count})."
            )

        if profile.children:
            if len(profile.children) != 2:
                raise RuntimeError(
                    "A bifurcation must have exactly two children."
                )

            if profile.children[1].internal_bifurcation:
                return "bifurcate_internal"

            return "bifurcate"

        if not profile.active:
            return "annihilate"

        return None

    def synthesize(self, max_steps=None):
        """
        Generate the morphology from the NeuriteProfile topology.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of synthesis sweeps.

        Returns
        -------
        Neurite
            Soma containing the generated RandomWalk tree.
        """
        if max_steps is not None:
            if not isinstance(max_steps, int) or isinstance(max_steps, bool):
                raise TypeError("max_steps must be an integer or None.")

            if max_steps < 0:
                raise ValueError("max_steps cannot be negative.")

        soma = Neurite(
            points=[self.origin.copy()],
            section_type="soma",
        )

        if self.root.section_type == "soma":
            primary_profiles = list(self.root.children)
        else:
            primary_profiles = [self.root]

        theta, phi = self._resolve_primary_angles(
            len(primary_profiles)
        )

        primary_directions = misc.sphere_surface_points(
            n=len(primary_profiles),
            theta=theta,
            phi=phi,
        )

        if self.axis_direction is not None:
            primary_directions = np.asarray([
                misc.AxialFrame.to_global(
                    direction,
                    self.axis_direction,
                )
                for direction in primary_directions
            ])

        primary_walks = []

        for profile, initial_direction in zip(
            primary_profiles,
            primary_directions,
        ):
            walk = RandomWalk(
                rng=self.rng,
                step_size=profile.step_size,
                origin=self.origin,
                initial_direction=initial_direction,
                elongation_bias=self.elongation_bias,
                bifurcation_bias=self.bifurcation_bias,
                centrifugal=self.centrifugal,
                parent=soma,
                section_type=profile.section_type,
                max_angle=self.max_angle,
                elongation_random_weight=self.elongation_random_weight,
                elongation_bias_weight=self.elongation_bias_weight,
            )

            primary_walks.append(walk)

        for r in soma.children:
            print(r)

        active_neurites = {}

        for profile, walk in zip(primary_profiles, primary_walks):
            active_neurites.setdefault(profile.order, []).append((profile, walk))

        step = 0

        while active_neurites:
            current_order = min(active_neurites)

            for _, walk in active_neurites[current_order]:
                walk.active = True
                
            while active_neurites[current_order]:
                if max_steps is not None and step >= max_steps:
                    return soma

                current_neurites = active_neurites[current_order]
                active_neurites[current_order] = []

                for neurite in current_neurites:
                    profile, walk = neurite

                    if not walk.active:
                        continue

                    event = self.next_event(neurite)

                    if event == "elongate":
                        walk.elongate()
                        active_neurites[current_order].append(neurite)

                    elif event == "bifurcate":
                        children = walk.bifurcate()

                        for child_profile, child_walk in zip(profile.children, children):
                            active_neurites[child_profile.order].append((child_profile, child_walk))

                    elif event == "bifurcate_internal":
                        children = walk.bifurcate_internal()
                        for child_profile, child_walk in zip(profile.children, children):
                            active_neurites.setdefault(child_profile.order, list()).append((child_profile, child_walk))

                    elif event == "annihilate":
                        walk.annihilate()

                    elif event is not None:
                        raise RuntimeError(f"Unknown synthesis event: {event!r}.")

                for _, walk in current_neurites:
                    if walk.pending_event:
                        walk.update_state()

                step += 1

            del active_neurites[current_order]

        return soma

    def describe(self):
        """Print synthesized and experimental topology statistics."""
        synthesized_primary_count = float(len(self.soma.children))
        primary_range = self.primary_count_range_constraint

        if primary_range is None:
            experimental_primary = "N/A"
        elif isinstance(primary_range, dict):
            experimental_primary = (
                f"{float(primary_range['min']):.1f}–"
                f"{float(primary_range['max']):.1f}"
            )
        else:
            experimental_primary = (
                f"{float(primary_range[0]):.1f}–"
                f"{float(primary_range[1]):.1f}"
            )

        print(
            "Initial primary dendrites: "
            f"synthesized={synthesized_primary_count:.1f}, "
            f"experimental={experimental_primary}"
        )

        synthesized_bifurcation_count = float(
            self.soma.bifurcation_count()
        )
        bifurcation_constraint = (
            self.bifurcation_count_constraint
        )

        if bifurcation_constraint is None:
            experimental_bifurcations = "N/A"
        else:
            experimental_bifurcations = (
                f"{float(bifurcation_constraint['mean']):.1f} ± "
                f"{float(bifurcation_constraint['std']):.1f}"
            )

        print(
            "Bifurcation count: "
            f"synthesized={synthesized_bifurcation_count:.1f}, "
            f"experimental={experimental_bifurcations}"
        )

        sholl_constraint = self.sholl_plot_constraint

        if sholl_constraint is None:
            synthesized_sholl = np.asarray(
                self.soma.sholl_plot(
                    bin_size=self.bin_size,
                ),
                dtype=float,
            )

            print("Sholl plot:")
            print("radius  synthesized")

            for index, synthesized in enumerate(
                synthesized_sholl
            ):
                radius = index * self.bin_size
                print(
                    f"{radius:.1f}  "
                    f"{synthesized:.1f}"
                )

            return

        experimental_mean = np.asarray(
            sholl_constraint["mean"],
            dtype=float,
        )
        experimental_std = np.asarray(
            sholl_constraint["std"],
            dtype=float,
        )

        if experimental_mean.ndim != 1 or experimental_std.shape != experimental_mean.shape:
            raise ValueError(
                "Sholl mean and standard deviation must be "
                "one-dimensional arrays with equal shape."
            )

        max_distance = (
            max(len(experimental_mean) - 1, 0)
            * self.bin_size
        )
        synthesized_sholl = np.asarray(
            self.soma.sholl_plot(
                bin_size=self.bin_size,
                max_distance=max_distance,
            ),
            dtype=float,
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
            radius = index * self.bin_size

            print(
                f"{radius:.1f}  "
                f"{synthesized:.1f}  "
                f"{mean:.1f}  "
                f"{std:.1f}"
            )
