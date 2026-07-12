import numpy as np

from .. import misc


class RandomWalk:
    """Represent one branching-annihilating random walk."""

    def __init__(
        self,
        rng,
        first_point,
        step_size,
        origin=None,
        initial_direction=None,
        elongation_bias=None,
        bifurcation_bias=None,
        centrifugal=False,
        parent=None,
        active=True,
    ):
        """Initialize the random walk."""
        if not hasattr(rng, "normal"):
            raise TypeError("rng must provide a normal() method.")

        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        if origin is None:
            origin = np.zeros(3, dtype=float)

        origin = self._validate_vector(
            origin,
            "origin",
        )
        first_point = self._validate_vector(
            first_point,
            "first_point",
        )

        if initial_direction is not None:
            initial_direction = misc._normalize(
                self._validate_vector(
                    initial_direction,
                    "initial_direction",
                )
            )

        self._validate_bias(
            elongation_bias,
            "elongation_bias",
        )
        self._validate_bias(
            bifurcation_bias,
            "bifurcation_bias",
        )

        self.rng = rng
        self.origin = origin.copy()
        self.step_size = float(step_size)

        self.initial_direction = (
            None
            if initial_direction is None
            else initial_direction.copy()
        )

        self.elongation_bias = elongation_bias
        self.bifurcation_bias = bifurcation_bias
        self.centrifugal = bool(centrifugal)

        self.parent = parent
        self.children = []

        self.points = [first_point.copy()]
        self.active = bool(active)
        self.internal_bifurcation = False

        self.pending_move = None

    @property
    def first_point(self):
        """Return the first point."""
        return self.points[0]

    @property
    def current_point(self):
        """Return the current endpoint."""
        return self.points[-1]

    @property
    def last_direction(self):
        """Return the latest accepted unit direction."""
        if len(self.points) >= 2:
            return misc._normalize(
                self.points[-1] - self.points[-2]
            )

        if self.initial_direction is None:
            return None

        return self.initial_direction.copy()

    def elongate(self):
        """Propose an elongation move."""
        self._check_move_allowed()

        bias = self._compute_elongation_bias()

        direction = self._generate_elongation_direction(
            bias
        )

        point = self._generate_point(direction)

        self.pending_move = {
            "event": "elongation",
            "point": point,
            "direction": direction,
        }

        return point

    def bifurcate(self):
        """Propose a bifurcation into two active children."""
        return self._propose_bifurcation(
            event="bifurcation"
        )

    def bifurcate_internal(self):
        """Propose one active and one inactive child."""
        return self._propose_bifurcation(
            event="internal_bifurcation"
        )

    def annihilate(self):
        """Propose annihilation."""
        self._check_move_allowed()

        self.pending_move = {
            "event": "annihilation",
        }

    def update_state(self):
        """Commit the pending move."""
        if self.pending_move is None:
            raise RuntimeError(
                "No pending move is available."
            )

        event = self.pending_move["event"]

        if event == "elongation":
            point = self.pending_move["point"].copy()

            self.points.append(point)
            result = point

        elif event == "bifurcation":
            children = self.pending_move["children"]
            points = self.pending_move["points"]

            for child, point in zip(children, points):
                child.points.append(point.copy())

            self.children = list(children)
            self.active = False
            self.internal_bifurcation = False

            result = tuple(children)

        elif event == "internal_bifurcation":
            children = self.pending_move["children"]
            points = self.pending_move["points"]

            for child, point in zip(children, points):
                child.points.append(point.copy())

            children[0].active = True
            children[1].active = False

            self.children = list(children)
            self.active = False
            self.internal_bifurcation = True

            result = tuple(children)

        elif event == "annihilation":
            self.active = False
            result = None

        else:
            raise RuntimeError(
                f"Unknown pending event: {event!r}."
            )

        self.pending_move = None

        return result

    def discard_pending_move(self):
        """Discard the pending move."""
        if self.pending_move is None:
            raise RuntimeError(
                "No pending move is available."
            )

        self.pending_move = None

    def activate_internal_branch(self):
        """Activate the inactive internal branch."""
        if not self.internal_bifurcation:
            raise RuntimeError(
                "This walk is not an internal bifurcation."
            )

        if len(self.children) != 2:
            raise RuntimeError(
                "An internal bifurcation must have two children."
            )

        branch = self.children[1]

        if branch.active:
            raise RuntimeError(
                "The internal branch is already active."
            )

        branch.active = True

        return branch

    def _propose_bifurcation(self, event):
        """Propose a standard or internal bifurcation."""
        self._check_move_allowed()

        directions = (
            self._compute_bifurcation_directions()
        )

        points = [
            self._generate_point(direction)
            for direction in directions
        ]

        children = [
            self._create_child(
                initial_direction=direction,
                active=True,
            )
            for direction in directions
        ]

        self.pending_move = {
            "event": event,
            "children": children,
            "points": points,
            "directions": directions,
        }

        return tuple(children)

    def _create_child(
        self,
        initial_direction,
        active,
    ):
        """Create a pending child random walk."""
        return self.__class__(
            rng=self.rng,
            first_point=self.current_point,
            step_size=self.step_size,
            origin=self.origin,
            initial_direction=initial_direction,
            elongation_bias=self.elongation_bias,
            bifurcation_bias=self.bifurcation_bias,
            centrifugal=self.centrifugal,
            parent=self,
            active=active,
        )

    def _compute_elongation_bias(self):
        """Return the elongation bias."""
        if self.elongation_bias is None:
            return None

        bias = self.elongation_bias.compute(
            self,
            self.current_point.copy(),
        )

        if bias is None:
            return None

        bias = np.asarray(
            bias,
            dtype=float,
        )

        if bias.shape != (3,):
            raise ValueError(
                "elongation_bias.compute() must return "
                "None or an array with shape (3,)."
            )

        return bias

    def _compute_bifurcation_directions(self):
        """Return two unit initial directions."""
        if self.bifurcation_bias is None:
            return np.asarray(
                [
                    self._sample_direction(),
                    self._sample_direction(),
                ]
            )

        directions = self.bifurcation_bias.compute(
            self,
            self.current_point.copy(),
        )

        if directions is None:
            return np.asarray(
                [
                    self._sample_direction(),
                    self._sample_direction(),
                ]
            )

        directions = np.asarray(
            directions,
            dtype=float,
        )

        if directions.shape != (2, 3):
            raise ValueError(
                "bifurcation_bias.compute() must return "
                "None or an array with shape (2, 3)."
            )

        return np.asarray(
            [
                misc._normalize(directions[0]),
                misc._normalize(directions[1]),
            ]
        )

    def _generate_elongation_direction(
        self,
        bias,
    ):
        """Generate a unit elongation direction."""
        direction = self.last_direction

        if direction is None:
            direction = self._sample_direction()

        if bias is not None:
            direction = direction + bias

        return misc._normalize(direction)

    def _generate_point(self, direction):
        """Generate a proposed point."""
        direction = misc._normalize(direction)

        if not self.centrifugal:
            return (
                self.current_point
                + self.step_size * direction
            )

        radial_direction = (
            self._centrifugal_direction()
        )

        tangential_component = (
            direction
            - np.dot(
                direction,
                radial_direction,
            )
            * radial_direction
        )

        displacement = (
            self.step_size * radial_direction
            + tangential_component
        )

        return self.current_point + displacement

    def _centrifugal_direction(self):
        """Return the outward unit direction from the origin."""
        radial_vector = (
            self.current_point - self.origin
        )

        if np.isclose(
            np.linalg.norm(radial_vector),
            0.0,
        ):
            if self.last_direction is not None:
                return self.last_direction

            return self._sample_direction()

        return misc._normalize(radial_vector)

    def _sample_direction(self):
        """Sample a random unit direction."""
        while True:
            vector = self.rng.normal(size=3)

            if not np.isclose(
                np.linalg.norm(vector),
                0.0,
            ):
                return misc._normalize(vector)

    def _check_move_allowed(self):
        """Check whether a move can be proposed."""
        if not self.active:
            raise RuntimeError(
                "An inactive random walk cannot propose a move."
            )

        if self.children:
            raise RuntimeError(
                "A random walk with children cannot propose a move."
            )

        if self.pending_move is not None:
            raise RuntimeError(
                f"A {self.pending_move['event']!r} move "
                "is already pending."
            )

    def _iter_walks(self):
        """Iterate over this walk and its descendants."""
        yield self

        for child in self.children:
            yield from child._iter_walks()

    @staticmethod
    def _validate_bias(bias, name):
        """Validate an optional bias object."""
        if (
            bias is not None
            and not callable(
                getattr(bias, "compute", None)
            )
        ):
            raise TypeError(
                f"{name} must provide a callable compute() method."
            )

    @staticmethod
    def _validate_vector(vector, name):
        """Validate a 3D vector."""
        vector = np.asarray(
            vector,
            dtype=float,
        )

        if vector.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,)."
            )

        return vector
