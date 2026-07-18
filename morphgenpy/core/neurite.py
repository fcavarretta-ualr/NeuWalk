import numpy as np

from .. import misc


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
        self._children = []

        if parent is not None:
            self.connect(parent, relation="parent")

    @property
    def children(self):
        """Return a copy of the child sections."""
        return self._children.copy()

    @property
    def length(self):
        """Return the length of the neurite."""
        points = np.asarray(self.points, dtype=float)

        return np.sum(
            np.linalg.norm(
                points[1:, :] - points[:-1, :],
                axis=1,
            )
        )

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
            len(neurite.children) == 2
            for neurite in self.subtree if neurite.section_type != "soma"
        )

    @property
    def total_length(self):
        """Return the total Euclidean length of this subtree."""
        return sum(
            np.linalg.norm(point_1 - point_0)
            for point_0, point_1 in self._segments()
        )

    @property
    def siblings(self):
        """Return sibling sections."""
        if not self.parent:
            return []

        siblings = set(self.parent.children)
        siblings.discard(self)

        return list(siblings)
    
    def disconnect_from_parent(self):
        """Disconnect this neurite from its parent and return the parent."""
        if self.parent:
            self.disconnect(self.parent)

    def disconnect_from_children(self):
        """Disconnect and return all child neurites."""
        for child in self.children:
            self.disconnect(child)
    
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
        if relation == "child":
            self._connect_child(neurite)
        elif relation == "parent":
            neurite._connect_child(self)
        else:
            raise Exception(f"Unknown relation {relation}")

    def disconnect(self, neurite=None):
        """
        Disconnect this neurite's parent or one of its children.

        Parameters
        ----------
        neurite : Neurite
            Specific neurite to disconnect.
        """
        if neurite is None:
            self.disconnect_from_parent()
            self.disconnect_from_children()
            return
        
        if neurite is self.parent:
            self.parent._children.remove(self)
            self.parent = None

        elif neurite in self._children:
            self._children.remove(neurite)
            neurite.parent = None

        else:
            raise ValueError(
                "The supplied neurite is not this neurite's parent or child."
            )

    def _connect_child(self, child):
        """Connect ``child`` directly below this neurite."""
        if not isinstance(child, Neurite):
            raise TypeError("neurite must be a Neurite.")

        if self in child.subtree or child in self.subtree:
            raise ValueError("Sections already connected.")

        if child.parent:
            raise RuntimeError("Child already connected.")

        child.parent = self
        self._children.append(child)

    def _merge_with_descendant(self):
        """
        Merge this section with its only child of the same type.

        The child's first point is assumed to coincide with this section's last
        point and is therefore not duplicated. The child's descendants are
        reconnected directly to this section.

        Returns
        -------
        Neurite
            The merged section.
        """
        if len(self._children) != 1:
            raise RuntimeError(
                "Merging requires exactly one child."
            )

        descendant = self._children[0]

        if self.section_type != descendant.section_type:
            raise RuntimeError(
                "Sections must have the same section_type."
            )

        if len(self.points) == 0 or len(descendant.points) == 0:
            raise RuntimeError(
                "Both sections must contain at least one point."
            )

        if not np.allclose(
            self.points[-1],
            descendant.points[0],
        ):
            raise ValueError(
                "The child must start at the parent endpoint."
            )

        self.points = np.vstack(
            (
                self.points,
                descendant.points[1:],
            )
        )

        grandchildren = descendant.children

        self.disconnect(descendant)

        for child in grandchildren:
            descendant.disconnect(child)
            self.connect(child, relation="child")

        return self

    def _event_counts(self, bin_size, max_distance=None):
        """
        Count terminal, bifurcation, and internal-branch events by spatial bin.

        The bins match the intervals used by ``sholl_plot``. Therefore, if the
        Sholl plot has length ``n``, each returned array has length ``n - 1``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(bifurcations, annihilations, internal_bifurcations)``.
        """
        sholl = self.sholl_plot(
            bin_size=bin_size,
            max_distance=max_distance,
        )
        n_bins = max(len(sholl) - 1, 0)

        bifurcations = np.zeros(n_bins, dtype=int)
        annihilations = np.zeros(n_bins, dtype=int)
        internal_bifurcations = np.zeros(n_bins, dtype=int)

        if self.section_type == "soma":
            subtrees = [
                child
                for child in self._children
                if len(child.points)
                and child.section_type != "unknown"
            ]

        elif len(self.points):
            subtrees = [self]

        else:
            return (
                bifurcations,
                annihilations,
                internal_bifurcations,
            )

        def bin_index(distance):
            if n_bins == 0 or distance < 0:
                return None

            # A point exactly on a boundary belongs to the preceding bin.
            adjusted = np.nextafter(distance, 0.0)
            index = int(np.floor(adjusted / bin_size))

            return index if 0 <= index < n_bins else None

        for subtree in subtrees:
            origin = subtree.points[0]

            for section in subtree.subtree:
                if (
                    section.section_type == "unknown"
                    or len(section.points) == 0
                ):
                    continue

                endpoint = misc.translate_points(
                    section.points[-1],
                    source=origin,
                )
                index = bin_index(np.linalg.norm(endpoint))

                if index is None:
                    continue

                valid_children = [
                    child
                    for child in section.children
                    if child.section_type != "unknown"
                ]

                if not valid_children:
                    annihilations[index] += 1
                    continue

                if len(valid_children) < 2:
                    continue

                is_internal = any(
                    child.section_type != self.section_type
                    for child in valid_children
                )

                if is_internal:
                    internal_bifurcations[index] += 1
                else:
                    bifurcations[index] += 1

        return (
            bifurcations,
            annihilations,
            internal_bifurcations,
        )

    def sholl_plot(self, bin_size, max_distance=None):
        """
        Calculate a Sholl plot after aligning primary dendrite origins.

        Bin zero contains the number of primary dendrites. Each primary
        dendrite subtree is translated so that its first point lies at the
        origin before shell intersections are calculated.
        """
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        # Standard neuron: the root is generally the soma and its children
        # are the primary dendrites.
        if self.section_type == "soma":
            primary_dendrites = [
                child
                for child in self._children
                if len(child.points) > 0
            ]

        # Disconnected morphology: the root itself represents one primary
        # dendrite.
        elif len(self.points) > 0:
            primary_dendrites = [self]

        else:
            return np.zeros(1, dtype=int)

        segments = []

    
        for primary in primary_dendrites:
            source = primary.points[0].copy()
            for section in primary.subtree:
                for point_0, point_1 in zip(section.points[:-1], section.points[1:]):
                    segments.append((point_0 - source, point_1 - source))

        if max_distance is None:
            max_distance = max(
                (
                    np.linalg.norm(point)
                    for segment in segments
                    for point in segment
                ),
                default=0.0,
            )

        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")
        
        radii = np.arange(
            0.0,
            max_distance + bin_size,
            bin_size,
        )

        crossings = np.zeros(len(radii), dtype=int)

        # This value is assigned independently of shell intersections.

        for point_0, point_1 in segments:
            distance_0 = np.linalg.norm(point_0)
            distance_1 = np.linalg.norm(point_1)
            start = min(distance_0, distance_1)
            end = max(distance_0, distance_1)
            crossings += (radii >= start) & (radii < end)

        return crossings.astype(int)

    def _traverse(self):
        """Yield this section and its descendants depth-first."""
        yield self

        for child in self._children:
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
                if not np.allclose(
                    parent_endpoint,
                    child_start,
                ):
                    yield parent_endpoint, child_start

    @staticmethod
    def _validate_points(points):
        """Return points as a float array with shape ``(n, 3)``."""
        if points is None:
            return []

        points = np.asarray(points, dtype=float)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3).")

        return [
            np.array(point, dtype=float)
            for point in points
        ]

    def clone(self):
        """Return an independent copy of this neurite subtree."""
        cloned = self.__class__(
            points=self.points.copy(),
            section_type=self.section_type,
        )

        for child in self._children:
            cloned.connect(
                child.clone(),
                relation="child",
            )

        return cloned
