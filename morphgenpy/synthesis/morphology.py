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
        theta : float or sequence of two floats, optional
            Polar angle or angular range used to generate the primary
            directions.
        phi : float or sequence of two floats, optional
            Azimuthal angle or angular range used to generate the primary
            directions.
        axis_direction : array_like, optional
            Direction of the local axial frame in global coordinates.
        elongation_bias : ElongationBias or sequence, optional
            Elongation bias or weighted elongation biases.
        bifurcation_bias : BifurcationBias, optional
            Bifurcation bias passed to each RandomWalk.
        centrifugal : bool, default False
            Whether RandomWalk displacement is centrifugal.
        max_angle : float, default pi / 2
            Maximum angle used when sampling elongation directions.
        elongation_random_weight : float, default 1.0
            Weight of the random elongation component.
        elongation_bias_weight : float, default 1.0
            Global weight applied to elongation biases.
        """
        if not isinstance(root, NeuriteProfile):
            raise TypeError(
                "root must be a NeuriteProfile."
            )

        if origin is None:
            origin = np.zeros(3, dtype=float)

        origin = np.asarray(origin, dtype=float)

        if origin.shape != (3,):
            raise ValueError(
                "origin must have shape (3,)."
            )

        if axis_direction is not None:
            axis_direction = np.asarray(
                axis_direction,
                dtype=float,
            )

            if axis_direction.shape != (3,):
                raise ValueError(
                    "axis_direction must have shape (3,)."
                )

            if np.isclose(
                np.linalg.norm(axis_direction),
                0.0,
            ):
                raise ValueError(
                    "axis_direction cannot be the zero vector."
                )

        self.root = root
        self.rng = rng
        self.origin = origin.copy()
        self.theta = theta
        self.phi = phi

        self.axis_direction = (
            None
            if axis_direction is None
            else axis_direction.copy()
        )

        self.elongation_bias = elongation_bias
        self.bifurcation_bias = bifurcation_bias
        self.centrifugal = bool(centrifugal)
        self.max_angle = max_angle
        self.elongation_random_weight = elongation_random_weight
        self.elongation_bias_weight = elongation_bias_weight

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

            if profile.internal_bifurcation:
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
            if (
                not isinstance(max_steps, int)
                or isinstance(max_steps, bool)
            ):
                raise TypeError(
                    "max_steps must be an integer or None."
                )

            if max_steps < 0:
                raise ValueError(
                    "max_steps cannot be negative."
                )

        soma = Neurite(
            points=[self.origin.copy()],
            section_type="soma",
        )

        if self.root.section_type == "soma":
            primary_profiles = list(self.root.children)
        else:
            primary_profiles = [self.root]

        primary_directions = misc.sphere_surface_points(
            n=len(primary_profiles),
            theta=self.theta,
            phi=self.phi,
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
        active_neurites = list(
            zip(
                primary_profiles,
                primary_walks,
            )
        )

        step = 0

        while active_neurites:
            if (
                max_steps is not None
                and step >= max_steps
            ):
                break

            next_active_neurites = []

            for neurite in active_neurites:
                profile, walk = neurite

                if not walk.active:
                    continue

                event = self.next_event(neurite)

                if event == "elongate":
                    walk.elongate()
                    next_active_neurites.append(neurite)

                elif event == "bifurcate":
                    children = walk.bifurcate()

                    for child_profile, child_walk in zip(
                        profile.children,
                        children,
                    ):
                        next_active_neurites.append(
                            (child_profile, child_walk)
                        )

                elif event == "bifurcate_internal":
                    children = walk.bifurcate_internal()

                    next_active_neurites.append(
                        (
                            profile.children[0],
                            children[0],
                        )
                    )

                elif event == "annihilate":
                    walk.annihilate()

                elif event is not None:
                    raise RuntimeError(
                        f"Unknown synthesis event: {event!r}."
                    )

            # update state for all
            for _, walk in active_neurites:
                walk.update_state()

            active_neurites = next_active_neurites
            step += 1

        return soma
