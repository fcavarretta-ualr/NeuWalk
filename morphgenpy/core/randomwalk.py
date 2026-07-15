import numpy as np

from .. import misc
from ..biases import ElongationBias
from .neurite import Neurite


class RandomWalk(Neurite):
    """Represent one branching-annihilating random walk."""

    _MIN_CENTRIFUGAL_ALIGNMENT = 1e-6

    def __init__(
        self,
        rng,
        step_size,
        first_point=None,
        origin=None,
        initial_direction=None,
        elongation_bias=None,
        bifurcation_bias=None,
        centrifugal=True,
        parent=None,
        active=True,
        section_type=None,
        max_angle=np.pi / 2,
        elongation_random_weight=0.01,
        elongation_random_hill_k=1.0,
        elongation_random_hill_n=-1.0,
        elongation_bias_weight=1.0,
    ):
        """Initialize the random walk."""

        if not np.isscalar(step_size):
            raise TypeError("step_size must be a scalar.")

        step_size = float(step_size)

        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be finite and positive.")

        origin = np.zeros(3, dtype=float) if origin is None else self._validate_vector(origin, "origin")

        if first_point is None:
            first_point = parent.points[-1].copy() if parent is not None else origin.copy()
        else:
            first_point = self._validate_vector(first_point, "first_point")

            if parent is not None and not np.allclose(first_point, parent.points[-1]):
                raise ValueError("first_point does not correspond to the last point of the parent.")

        first_point = self._validate_vector(first_point, "first_point")

        if initial_direction is None and parent is None:
            raise ValueError("initial_direction is required for the root random walk.")

        super().__init__(points=[first_point], section_type=section_type, parent=parent)

        if initial_direction is not None:
            initial_direction = misc._normalize(self._validate_vector(initial_direction, "initial_direction"))

        self.elongation_biases = self._prepare_elongation_biases(elongation_bias)
        self._validate_bias(bifurcation_bias, "bifurcation_bias")

        self.rng = rng
        self.origin = origin.copy()
        self.step_size = step_size
        self.initial_direction = None if initial_direction is None else initial_direction.copy()
        self.bifurcation_bias = bifurcation_bias
        self.centrifugal = bool(centrifugal)
        self.active = bool(active)
        self.internal_bifurcation = False
        self.pending_event = None

        if not np.isscalar(max_angle):
            raise TypeError("max_angle must be a scalar.")

        max_angle = float(max_angle)

        if not np.isfinite(max_angle) or not 0.0 <= max_angle <= np.pi:
            raise ValueError("max_angle must be finite and lie within [0, pi].")

        self.max_angle = max_angle

        if not np.isscalar(elongation_random_weight):
            raise TypeError("elongation_random_weight must be a scalar.")

        elongation_random_weight = float(elongation_random_weight)

        if not np.isfinite(elongation_random_weight) or elongation_random_weight < 0.0:
            raise ValueError("elongation_random_weight must be finite and non-negative.")

        self.elongation_random_weight = elongation_random_weight

        if not np.isscalar(elongation_random_hill_k):
            raise TypeError("elongation_random_hill_k must be a scalar.")

        elongation_random_hill_k = float(elongation_random_hill_k)

        if not np.isfinite(elongation_random_hill_k) or elongation_random_hill_k <= 0.0:
            raise ValueError("elongation_random_hill_k must be finite and greater than 0.")

        self.elongation_random_hill_k = elongation_random_hill_k

        if not np.isscalar(elongation_random_hill_n):
            raise TypeError("elongation_random_hill_n must be a scalar.")

        elongation_random_hill_n = float(elongation_random_hill_n)

        if not np.isfinite(elongation_random_hill_n) or elongation_random_hill_n >= 0.0:
            raise ValueError("elongation_random_hill_n must be finite and less than 0.")

        self.elongation_random_hill_n = elongation_random_hill_n

        if not np.isscalar(elongation_bias_weight):
            raise TypeError("elongation_bias_weight must be a scalar.")

        elongation_bias_weight = float(elongation_bias_weight)

        if not np.isfinite(elongation_bias_weight) or elongation_bias_weight < 0.0:
            raise ValueError("elongation_bias_weight must be finite and non-negative.")

        self.elongation_bias_weight = elongation_bias_weight

    @property
    def first_point(self):
        """Return the first point."""
        return self.points[0].copy()

    @property
    def current_point(self):
        """Return the current endpoint."""
        return self.points[-1].copy()

    @property
    def last_direction(self):
        """Return the latest accepted unit direction."""
        if len(self.points) >= 2:
            return misc._normalize(self.points[-1] - self.points[-2])

        return self.initial_direction.copy()

    def _centrifugal_direction(self):
        """Return the outward unit direction from the origin."""
        if not self.centrifugal:
            raise RuntimeError("The centrifugal direction is unavailable when centrifugal=False.")

        displacement = self.current_point - self.origin

        if np.isclose(np.linalg.norm(displacement), 0.0):
            return self.initial_direction.copy()

        return misc._normalize(displacement)

    def _step_size(self, direction):
        """Return the step length accounting for the centrifugal component."""
        if not self.centrifugal:
            return self.step_size

        direction = misc._normalize(direction)
        alignment = float(np.dot(direction, self._centrifugal_direction()))

        if alignment <= self._MIN_CENTRIFUGAL_ALIGNMENT:
            raise ValueError("direction must have a sufficiently positive centrifugal component.")

        return self.step_size / alignment

    def _generate_point(self, direction):
        """Generate a proposed point."""
        direction = misc._normalize(direction)
        return self.current_point + self._step_size(direction) * direction

    def _sample_direction(self, reference_direction):
        """Sample a random unit direction."""
        return misc.random_cone_direction(self.rng, reference_direction, self.max_angle)

    def elongate(self):
        """Propose an elongation move."""
        self._check_move_allowed()

        direction = self.last_direction
        step_size = self._step_size(direction)

        if not np.isclose(np.linalg.norm(self.first_point - self.origin), 0.0):
            for weight, bias in self.elongation_biases:
                value = bias.compute(self, direction)

                if value is None:
                    continue

                value = np.asarray(value, dtype=float)

                if value.shape != (3,):
                    raise ValueError("elongation_bias.compute() must return None or an array with shape (3,).")

                if not np.all(np.isfinite(value)):
                    raise ValueError("elongation_bias.compute() must return only finite values.")

                direction = misc._normalize(
                    direction + value * weight * self.elongation_bias_weight * step_size
                )

                step_size = self._step_size(direction)

        hill_value = misc.hill(
            step_size,
            self.elongation_random_hill_k,
            self.elongation_random_hill_n,
        )

        if not np.isscalar(hill_value):
            raise TypeError("misc.hill() must return a scalar.")

        hill_value = float(hill_value)

        if not np.isfinite(hill_value):
            raise ValueError("misc.hill() must return a finite value.")

        random_component = self._sample_direction(direction) * hill_value
        direction = misc._normalize(direction + random_component * self.elongation_random_weight)

        if self.centrifugal:
            centrifugal_direction = self._centrifugal_direction()

            if np.dot(direction, centrifugal_direction) <= 0.0:
                direction = centrifugal_direction

        point = self._generate_point(direction)
        self.pending_event = {"event": "elongation", "point": point, "direction": direction}

        return point

    def bifurcate(self):
        """Propose a bifurcation into two active children."""
        return self._propose_bifurcation(event="bifurcation")

    def bifurcate_internal(self):
        """Propose one active and one inactive child."""
        return self._propose_bifurcation(event="internal_bifurcation")

    def annihilate(self):
        """Propose annihilation."""
        self._check_move_allowed()
        self.pending_event = {"event": "annihilation"}

    def update_state(self):
        """Commit the pending move."""
        if self.pending_event is None:
            raise RuntimeError("No pending move is available.")

        event = self.pending_event["event"]

        if event == "elongation":
            point = self.pending_event["point"].copy()
            self.points.append(point)
            result = point

        elif event == "bifurcation":
            children = self.pending_event["children"]
            self._children = list(children)
            self.active = False
            self.internal_bifurcation = False
            result = tuple(children)

        elif event == "internal_bifurcation":
            children = self.pending_event["children"]
            points = self.pending_event["points"]

            for child, point in zip(children, points):
                child.points.append(point.copy())

            children[0].active = True
            children[1].active = False
            self._children = list(children)
            self.active = False
            self.internal_bifurcation = True
            result = tuple(children)

        elif event == "annihilation":
            self.active = False
            result = None

        else:
            raise RuntimeError(f"Unknown pending event: {event!r}.")

        self.pending_event = None
        return result

    def discard_pending_event(self):
        """Discard the pending move."""
        if self.pending_event is None:
            raise RuntimeError("No pending move is available.")

        self.pending_event = None

    def activate_internal_branch(self):
        """Activate the inactive internal branch."""
        if not self.internal_bifurcation:
            raise RuntimeError("This walk is not an internal bifurcation.")

        if len(self._children) != 2:
            raise RuntimeError("An internal bifurcation must have two children.")

        branch = self._children[1]

        if branch.active:
            raise RuntimeError("The internal branch is already active.")

        branch.active = True
        return branch

    def _propose_bifurcation(self, event):
        """Propose a standard or internal bifurcation."""
        self._check_move_allowed()

        directions = self._compute_bifurcation_directions()
        children = [self._create_child(direction, active=True) for direction in directions]
        pending_event = {"event": event, "children": children, "directions": directions}

        if event == "internal_bifurcation":
            pending_event["points"] = [
                child._generate_point(direction)
                for child, direction in zip(children, directions)
            ]

        self.pending_event = pending_event
        return tuple(children)

    def _create_child(self, initial_direction, active):
        """Create a pending child random walk."""
        return self.__class__(
            rng=self.rng,
            step_size=self.step_size,
            first_point=self.current_point,
            origin=self.origin,
            initial_direction=initial_direction,
            elongation_bias=self.elongation_biases,
            bifurcation_bias=self.bifurcation_bias,
            centrifugal=self.centrifugal,
            parent=self,
            active=active,
            section_type=self.section_type,
            max_angle=self.max_angle,
            elongation_random_weight=self.elongation_random_weight,
            elongation_random_hill_k=self.elongation_random_hill_k,
            elongation_random_hill_n=self.elongation_random_hill_n,
            elongation_bias_weight=self.elongation_bias_weight,
        )

    def _compute_bifurcation_directions(self):
        """Return two unit initial directions."""
        if self.bifurcation_bias is None:
            return np.asarray([
                self._sample_direction(self.last_direction),
                self._sample_direction(self.last_direction),
            ])

        directions = self.bifurcation_bias.compute(self)

        if directions is None:
            return np.asarray([
                self._sample_direction(self.last_direction),
                self._sample_direction(self.last_direction),
            ])

        directions = np.asarray(directions, dtype=float)

        if directions.shape != (2, 3):
            raise ValueError("bifurcation_bias.compute() must return None or an array with shape (2, 3).")

        if not np.all(np.isfinite(directions)):
            raise ValueError("bifurcation_bias.compute() must return only finite values.")

        return np.asarray([misc._normalize(directions[0]), misc._normalize(directions[1])])

    def _check_move_allowed(self):
        """Check whether a move can be proposed."""
        if not self.active:
            raise RuntimeError("An inactive random walk cannot propose a move.")

        if self._children:
            raise RuntimeError("A random walk with children cannot propose a move.")

        if self.pending_event is not None:
            raise RuntimeError(f"A {self.pending_event['event']!r} move is already pending.")

    def _iter_walks(self):
        """Iterate over this walk and its descendants."""
        yield self

        for child in self._children:
            yield from child._iter_walks()

    @staticmethod
    def _prepare_elongation_biases(biases):
        """Return elongation biases as weighted pairs."""
        if biases is None:
            return []

        if isinstance(biases, ElongationBias):
            biases = [(1.0, biases)]

        result = []

        for weight, bias in biases:
            if not isinstance(bias, ElongationBias):
                raise TypeError("Each elongation bias must be an ElongationBias.")

            if not np.isscalar(weight):
                raise TypeError("Elongation bias weights must be scalars.")

            weight = float(weight)

            if not np.isfinite(weight):
                raise ValueError("Elongation bias weights must be finite.")

            result.append((weight, bias))

        return result

    @staticmethod
    def _validate_bias(bias, name):
        """Validate an optional bias object."""
        if bias is not None and not callable(getattr(bias, "compute", None)):
            raise TypeError(f"{name} must provide a callable compute() method.")

    @staticmethod
    def _validate_vector(vector, name):
        """Validate a finite 3D vector."""
        vector = np.asarray(vector, dtype=float)

        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,).")

        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")

        return vector

    def to_neurite(self):
        """Convert this RandomWalk tree to a Neurite tree."""
        neurite = Neurite(points=np.asarray(self.points, dtype=float).copy(), section_type=self.section_type)

        for child in self._children:
            neurite.connect(child.to_neurite(), relation="child")

        return neurite
