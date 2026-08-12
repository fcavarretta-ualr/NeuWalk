import numpy as np
from .section_object import SectionObject, TYPE_LABELS, TYPE_CODES
from .. import misc

def is_float_array(value):
    return isinstance(value, np.ndarray) and value.shape == (3,) and np.issubdtype(value.dtype, np.floating)

        
class ValidatedList(list):
    def __init__(self, values=(), validator=None):
        self.validator = validator
        super().__init__()
        self.extend(values)

    def _validate(self, value):
        if self.validator is not None and not self.validator(value):
            raise TypeError(f"Invalid value: {value!r}")
        return value

    def append(self, value):
        super().append(self._validate(value))

    def extend(self, values):
        super().extend(self._validate(value) for value in values)

    def insert(self, index, value):
        super().insert(index, self._validate(value))

    def __setitem__(self, index, value):
        value = [self._validate(item) for item in value] if isinstance(index, slice) else self._validate(value)
        super().__setitem__(index, value)
        
    def __getitem__(self, index):
        return super().__getitem__(index)

    def __iadd__(self, values):
        self.extend(values)
        return self

    def copy(self):
        return type(self)(self, validator=self.validator)
    
    
class Section(SectionObject):
    """Represent one section of a rooted tree."""

    def __init__(self, points=None, label=None, parent=None):
        """
        Parameters
        ----------
        points : array-like, optional
            Section points with shape ``(n, 3)``.
        label : object, optional
            Identifier describing the label.
        parent : Section, optional
            Parent section.
        """
        super().__init__(label=label, parent=parent)

        self._points = ValidatedList(validator=is_float_array)

        if points is not None:
            self.points = points

        
    @property
    def points(self):
        return self._points


    @points.setter
    def points(self, values):
        self._points = ValidatedList(values, validator=is_float_array)


    @property
    def length(self):
        """Return the length of the section."""
        points = np.asarray(self.points, dtype=float)

        return np.sum(
            np.linalg.norm(
                points[1:, :] - points[:-1, :],
                axis=1,
            )
        )







    def _event_counts(self, bin_size, max_distance=None):
        """
        Count terminal, bifurcation, and internal-branch events by spatial bin.

        Bin ``i`` covers the interval ``[i * bin_size, (i + 1) * bin_size)``,
        matching the bins used by ``sholl_plot``: for the same ``bin_size``
        and ``max_distance``, both methods return arrays of the same length,
        and bin ``i`` in one refers to the same spatial interval as bin ``i``
        in the other.

        Returns
        -------
        tuple of numpy.ndarray
            ``(bifurcations, annihilations, internal_bifurcations)``.
        """


        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        # Standard neuron: the root is generally the soma and its children
        # are the primary sections.
        # discard soma, unknown, or sections with one point only
        sections = {section for section in self.subtree if section.label not in {"soma", "unknown"} and len(section.points) > 1}

        # position from which calculate distance
        source = self.root.points[0]

        # create the vector counters
        if max_distance is None:
            max_distance = self._calculate_max_distance(bin_size)

        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")
        

        # create the histograms
        n_bins = int(max_distance / bin_size)+1
        
        bifurcations = np.zeros(n_bins, dtype=int)
        annihilations = np.zeros(n_bins, dtype=int)
        internal_bifurcations = np.zeros(n_bins, dtype=int)


        # go over all sections
        for section in sections:
            # clamp so a section ending past max_distance still lands in
            # the last bin rather than indexing out of bounds
            i_bin = min(int(np.linalg.norm(section.points[-1] - source) / bin_size), n_bins - 1)
            match len(section.children):
                case 0:
                    annihilations[i_bin] += 1
                case 2:
                    same_category_cnt = sum(section.label == ch.label for ch in section.children)

                    match same_category_cnt:
                        case 1:
                            internal_bifurcations[i_bin] += 1
                        case 2:
                            bifurcations[i_bin] += 1
                        case _:
                            raise ValueError("A section have both children of different types.")
                case 1:
                    pass
                case _:
                    raise ValueError("A section have more then two children.")
        return bifurcations, annihilations, internal_bifurcations


    def _calculate_max_distance(self, bin_size):
        # position from which calculate distance
        source = self.root.points[0]

        distances = []
        for section in self.subtree:
            if section.label != "soma":
                for point in section.points:
                    distances.append(
                        np.linalg.norm(point - source)
                        )
        return max(distances)
    

    def sholl_plot(self, bin_size, max_distance=None):
        """
        Calculate a Sholl plot after aligning primary section origins.

        Bin ``i`` counts the number of sections whose path passes through
        the interval ``[i * bin_size, (i + 1) * bin_size)`` at least once,
        determined from the section's full range of distances from the
        root (its minimum to its maximum), not just the direction of
        individual segments. This means a section that moves inward at
        some point along its path (e.g. due to a random elongation
        component) is still counted at every interval its path actually
        passes through, and is counted at most once per interval even if
        it revisits that interval multiple times.

        These are the same bins used by ``_event_counts``: for the same
        ``bin_size`` and ``max_distance``, both methods return arrays of
        the same length, with bin ``i`` referring to the same spatial
        interval in both.
        """
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        # Standard neuron: the root is generally the soma and its children
        # are the primary sections.
        # discard soma, unknown, or sections with one point only
        sections = {section for section in self.subtree if section.label not in {"soma", "unknown"} and len(section.points) > 1}

        # position from which calculate distance
        source = self.root.points[0]

        # create the vector counters
        if max_distance is None:
            max_distance = self._calculate_max_distance(bin_size)

        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        
        n_bins = int(max_distance / bin_size) + 1
        crossings = np.zeros(n_bins, dtype=int)

        # each section contributes at most once to a bin: count it in
        # every interval its path's distance-from-source range spans,
        # regardless of whether it moves inward or outward to get there
        for section in sections:
            if section.parent is None or section.parent.label == "soma":
                crossings[0] += 1
                
            distances = np.linalg.norm(np.asarray(section.points, dtype=float) - source, axis=1)

            min_bin = int(distances.min() / bin_size) + 1
            max_bin = min(int(distances.max() / bin_size) + 1, n_bins - 1)

            crossings[min_bin:max_bin] += 1

        return crossings





class Neuron(list):
    """List containing only Section objects."""

    def __init__(self, sections=()):
        super().__init__()
        self.extend(sections)

    @staticmethod
    def _check(section):
        """Validate a section."""
        if not isinstance(section, Section):
            raise TypeError(f"Expected Section, got {type(section).__name__}.")
        return section

    def append(self, section):
        super().append(self._check(section))

    def extend(self, sections):
        super().extend(self._check(section) for section in sections)

    def insert(self, index, section):
        super().insert(index, self._check(section))

    def __setitem__(self, index, value):
        value = [self._check(section) for section in value] if isinstance(index, slice) else self._check(value)
        super().__setitem__(index, value)

    def __iadd__(self, sections):
        self.extend(sections)
        return self

    def wholetree(self, label=None):
        """Iterate over all sections, optionally filtered by label."""
        for section in self:
            for descendant in section.wholetree():
                if label is None or descendant.label == label:
                    yield descendant
