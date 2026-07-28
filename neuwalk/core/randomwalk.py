import numpy as np

from .. import misc
from ..biases import ElongationBias
from .neurite import Neurite
import inspect

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
        bifurcation_internal_bias=None,
        centrifugal=True,
        parent=None,
        active=True,
        section_type=None,
        max_angle=np.pi / 2,
        elongation_random_weight=0.0,
        elongation_random_hill_k=None,
        elongation_random_hill_n=None,
        elongation_bias_weight=1.0,
        max_step_size=20
    ):
        """Initialize the random walk."""

        if not np.isscalar(step_size):
            raise TypeError("step_size must be a scalar.")

        step_size = float(step_size)

        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be finite and positive.")



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
            initial_direction = misc.to_unit_vector(self._validate_vector(initial_direction, "initial_direction"))

        self._elongation_bias = None
        self._bifurcation_bias = None
        self._bifurcation_internal_bias = None

        self.elongation_bias = elongation_bias
        self.bifurcation_bias = bifurcation_bias
        self.bifurcation_internal_bias = bifurcation_internal_bias

        self.rng = rng

        self.step_size = step_size
        self.initial_direction = None if initial_direction is None else initial_direction.copy()
        self.centrifugal = bool(centrifugal)
        self.active = bool(active)
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

        if elongation_random_hill_k is None:
            elongation_random_hill_k = self.step_size
            
        if not np.isscalar(elongation_random_hill_k):
            raise TypeError("elongation_random_hill_k must be a scalar.")

        elongation_random_hill_k = float(elongation_random_hill_k)

        if not np.isfinite(elongation_random_hill_k) or elongation_random_hill_k <= 0.0:
            raise ValueError("elongation_random_hill_k must be finite and greater than 0.")

        self.elongation_random_hill_k = elongation_random_hill_k

        if elongation_random_hill_n is None:
            elongation_random_hill_n = -1
            
        if not np.isscalar(elongation_random_hill_n):
            raise TypeError("elongation_random_hill_n must be a scalar.")

        elongation_random_hill_n = float(elongation_random_hill_n)

        if not np.isfinite(elongation_random_hill_n) or elongation_random_hill_n >= 0.0:
            raise ValueError("elongation_random_hill_n must be finite and less than 0.")

        self.elongation_random_hill_n = float(elongation_random_hill_n)

        if not np.isscalar(elongation_bias_weight):
            raise TypeError("elongation_bias_weight must be a scalar.")

        elongation_bias_weight = float(elongation_bias_weight)

        if not np.isfinite(elongation_bias_weight) or elongation_bias_weight < 0.0:
            raise ValueError("elongation_bias_weight must be finite and non-negative.")

        self.elongation_bias_weight = elongation_bias_weight
        
        if not np.isfinite(max_step_size) or step_size <= 0.0:
            raise ValueError("max_step_size must be finite and positive.")
        
        self.max_step_size = max_step_size


    @property
    def origin(self):
        return self.root.points[0].copy()

    @property
    def elongation_bias(self):
        return self._elongation_bias

    @elongation_bias.setter
    def elongation_bias(self, bias):
        self._elongation_bias = self._prepare_elongation_biases(bias)

    @property
    def bifurcation_internal_bias(self):
        return self._bifurcation_internal_bias

    @bifurcation_internal_bias.setter
    def bifurcation_internal_bias(self, bias):
        self._validate_bias(bias, "bifurcation_internal_bias")
        self._bifurcation_internal_bias = bias
        
    @property
    def bifurcation_bias(self):
        return self._bifurcation_bias

    @bifurcation_bias.setter
    def bifurcation_bias(self, bias):
        self._validate_bias(bias, "bifurcation_bias")
        self._bifurcation_bias = bias        

        
        
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
            return misc.to_unit_vector(self.points[-1] - self.points[-2])

        return self.initial_direction.copy()

    def _centrifugal_direction(self):
        """Return the outward unit direction from the origin."""

        displacement = self.current_point - self.origin

        if np.isclose(np.linalg.norm(displacement), 0.0):
            return self.initial_direction.copy()

        return misc.to_unit_vector(displacement)

    def _step_size(self, direction):
        """Return the step length accounting for the centrifugal component."""
        if not self.centrifugal:
            return self.step_size

        direction = misc.to_unit_vector(direction)
        alignment = np.dot(direction, self._centrifugal_direction())


        return min(self.step_size / alignment, self.max_step_size)

    def _generate_point(self, direction):
        """Generate a proposed point."""
        direction = misc.to_unit_vector(direction)
        return self.current_point + self._step_size(direction) * direction

    def _sample_direction(self, reference_direction):
        """Sample a random unit direction."""
        return misc.random_cone_direction(self.rng, reference_direction, self.max_angle)

        
    def elongate(self):
        """Propose an elongation move."""
        self._check_move_allowed()

        direction = self.last_direction
        step_size = self._step_size(direction)
        
        # if it is the first point, do not compute bias
        if not ( (self.parent is None or self.parent.section_type == "soma") and len(self.points) < 2 ):
            # calculate the effect of the bias
            for weight, bias in self.elongation_bias:
                value = bias.compute(self.rng, self, direction)

                if value is None:
                    continue

                value = np.asarray(value, dtype=float)

                if value.shape != (3,):
                    raise ValueError("elongation_bias.compute() must return None or an array with shape (3,).")

                if not np.all(np.isfinite(value)):
                    raise ValueError("elongation_bias.compute() must return only finite values.")

                direction = misc.to_unit_vector(
                    direction + value * weight * self.elongation_bias_weight * step_size
                )

                step_size = self._step_size(direction)


        # random component
        hill_value = misc.hill(
            step_size,
            self.elongation_random_hill_k,
            self.elongation_random_hill_n,
        )

        random_component = self._sample_direction(direction) * hill_value
        direction = misc.to_unit_vector(direction + random_component * self.elongation_random_weight)

        # check for centrifugal component
        # if it is null, then correct the direction
        if self.centrifugal:
            centrifugal_direction = self._centrifugal_direction()

            if np.dot(direction, centrifugal_direction) <= 0.0:
                direction = centrifugal_direction

        point = self._generate_point(direction)
        self.pending_event = {"event": "elongation", "point": point, "direction": direction}
        return point


    def bifurcate_internal(self):
        return self._bifurcate(True)

    
    def bifurcate(self):
        return self._bifurcate(False)


    def _bifurcate(self, internal=False):
        """Propose a bifurcation into two active children."""
        self._check_move_allowed()

        bias, event = (self.bifurcation_internal_bias, "bifurcation_internal") if internal else (self.bifurcation_bias, "bifurcation")
            
        if bias:
            initial_directions = bias.compute(self.rng, self)

        else:
            initial_directions = (
                self._sample_direction(self.last_direction),
                self._sample_direction(self.last_direction)
                )

    
        children = []

        for initial_direction in initial_directions:
            children.append(
                self._mk_child(initial_direction=initial_direction)
                )

        self.pending_event = {
            "event": event,
            "children": children
            }
        
        return children
    

    def _mk_child(self, **kwargs):
        parameters = inspect.signature(type(self).__init__).parameters
        values = {name: getattr(self, name) for name in parameters if name != "self"}

        # typical for children
        values['first_point'] = self.points[-1].copy()
        values['parent'] = self

        unknown = set(kwargs) - set(values)
        if unknown:
            raise TypeError(f"Unexpected parameter(s): {sorted(unknown)}")

        values.update(kwargs)
        return type(self)(**values)

    
    def annihilate(self):
        """Propose annihilation."""
        self._check_move_allowed()
        self.pending_event = {"event": "annihilation"}

    def update_state(self):
        """Commit the pending move."""
        if self.pending_event is None:
            raise RuntimeError("No pending move is available.")

        event = self.pending_event["event"]

        match event:
            case "elongation":
                point = self.pending_event["point"].copy()
                self.points.append(point)
                result = point

            case "bifurcation" | "bifurcation_internal":
                children = self.pending_event["children"]
                self._children = list(children)
                self.active = False
                result = tuple(children)

            case "annihilation":
                self.active = False
                result = None

            case _:
                raise RuntimeError(f"Unknown pending event: {event!r}.")

        self.pending_event = None
        return result

    def discard_pending_event(self):
        """Discard the pending move."""
        if self.pending_event is None:
            raise RuntimeError("No pending move is available.")

        self.pending_event = None



    def _check_move_allowed(self):
        """Check whether a move can be proposed."""
        if not self.active:
            raise RuntimeError("An inactive random walk cannot propose a move.")

        if self._children:
            raise RuntimeError("A random walk with children cannot propose a move.")

        if self.pending_event is not None:
            raise RuntimeError(f"A {self.pending_event['event']!r} move is already pending.")


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
