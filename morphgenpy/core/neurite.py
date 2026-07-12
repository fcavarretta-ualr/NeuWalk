import numpy as np


class Neurite:
    """Represent one section of a rooted neurite tree."""

    def __init__(
        self,
        points=None,
        section_type=None,
        parent=None,
    ):
        """
        Parameters
        ----------
        points : array-like, optional
            Section points with shape ``(n, 3)``.
        section_type : object, optional
            Identifier describing the section type.
        parent : Neurite, optional
            Parent section.
        """
        self.points = self._validate_points(points)
        self.section_type = section_type
        self.parent = None
        self.children = []

        if parent is not None:
            self.connect(parent, relation="parent")

    @property
    def root(self):
        """Return the root section of the tree."""
        neurite = self

        while neurite.parent is not None:
            neurite = neurite.parent

        return neurite

    @property
    def depth(self):
        """Return the number of connections from this section to the root."""
        depth = 0
        neurite = self

        while neurite.parent is not None:
            depth += 1
            neurite = neurite.parent

        return depth

    @property
    def subtree(self):
        """Return this section and all its descendants."""
        return list(self._traverse())

    @property
    def wholetree(self):
        """Return all sections belonging to the same tree."""
        return self.root.subtree

    @property
    def bifurcation_count(self):
        """Return the number of branching sections in this subtree."""
        return sum(
            len(neurite.children) >= 2
            for neurite in self.subtree
        )

    @property
    def total_length(self):
        """Return the total Euclidean length of this subtree."""
        return sum(
            np.linalg.norm(point_1 - point_0)
            for point_0, point_1 in self._segments()
        )

    def connect(self, neurite, relation="parent"):
        """
        Connect another neurite as this neurite's parent or child.

        Parameters
        ----------
        neurite : Neurite
            Neurite to connect.
        relation : {"parent", "child"}, default "parent"
            Relationship of ``neurite`` relative to this neurite.
        """
        relation = self._validate_relation(relation)

        if relation == "child":
            return self._connect_child(neurite)

        neurite._connect_child(self)
        return neurite

    def disconnect(self, neurite=None, relation=None):
        """
        Disconnect this neurite's parent or one of its children.

        Parameters
        ----------
        neurite : Neurite, optional
            Specific neurite to disconnect.
        relation : {"parent", "child"}
            Relationship of ``neurite`` relative to this neurite.
        """
        relation = self._validate_relation(relation)

        if relation == "parent":
            if self.parent is None:
                return None

            if neurite is not None and neurite is not self.parent:
                raise ValueError(
                    "The supplied neurite is not this neurite's parent."
                )

            parent = self.parent
            parent.children.remove(self)
            self.parent = None
            return parent

        if neurite is None:
            raise ValueError(
                "neurite must be provided when disconnecting a child."
            )

        if neurite.parent is not self or neurite not in self.children:
            raise ValueError(
                "The supplied neurite is not connected as a child."
            )

        self.children.remove(neurite)
        neurite.parent = None
        return neurite

    def _connect_child(self, child):
        """Connect ``child`` directly below this neurite."""
        if not isinstance(child, Neurite):
            raise TypeError("neurite must be a Neurite.")

        if child is self:
            raise ValueError("A neurite cannot be connected to itself.")

        if self in child.subtree:
            raise ValueError("The connection would create a cycle.")

        if child.parent is self:
            if child not in self.children:
                self.children.append(child)
            return child

        if child in self.children:
            raise RuntimeError(
                "Inconsistent topology: child is already listed but "
                "does not reference this neurite as its parent."
            )

        if child.parent is not None:
            child.disconnect(
                neurite=child.parent,
                relation="parent",
            )

        child.parent = self
        self.children.append(child)
        return child

    def sholl_plot(
        self,
        bin_size,
        max_distance=None,
        center=None,
    ):
        """
        Count subtree intersections with concentric spherical shells.
        """
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        if center is None:
            if len(self.root.points) == 0:
                raise ValueError("The root neurite has no points.")

            center = self.root.points[0]

        center = np.asarray(center, dtype=float)

        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")

        if max_distance is None:
            point_sets = [
                neurite.points
                for neurite in self.subtree
                if len(neurite.points)
            ]

            if not point_sets:
                return np.zeros(0, dtype=int)

            max_distance = np.max(
                np.linalg.norm(
                    np.vstack(point_sets) - center,
                    axis=1,
                )
            )

        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        radii = np.arange(
            0.0,
            max_distance + 0.5 * bin_size,
            bin_size,
        )
        counts = np.zeros(len(radii), dtype=int)

        for point_0, point_1 in self._segments():
            distance_0 = np.linalg.norm(point_0 - center)
            distance_1 = np.linalg.norm(point_1 - center)
            lower, upper = sorted((distance_0, distance_1))

            counts += (
                (radii > lower) & (radii <= upper)
            ).astype(int)

        return counts

    def _traverse(self):
        """Yield this section and its descendants depth-first."""
        yield self

        for child in self.children:
            yield from child._traverse()

    def _segments(self):
        """Yield each geometric segment in the subtree exactly once."""
        for neurite in self.subtree:
            for point_0, point_1 in zip(
                neurite.points[:-1],
                neurite.points[1:],
            ):
                yield point_0, point_1

            if (
                neurite.parent is not None
                and len(neurite.parent.points)
                and len(neurite.points)
            ):
                parent_endpoint = neurite.parent.points[-1]
                child_start = neurite.points[0]

                # Do not add an extra connector when the two sections
                # already share the same point.
                if not np.allclose(parent_endpoint, child_start):
                    yield parent_endpoint, child_start

    @staticmethod
    def _validate_relation(relation):
        """Validate a parent-child relation identifier."""
        if relation not in {"parent", "child"}:
            raise ValueError(
                "relation must be either 'parent' or 'child'."
            )

        return relation

    @staticmethod
    def _validate_points(points):
        """Return points as a float array with shape ``(n, 3)``."""
        if points is None:
            return np.empty((0, 3), dtype=float)

        points = np.asarray(points, dtype=float)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3).")

        return points.copy()
