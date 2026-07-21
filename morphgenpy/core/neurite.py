import numpy as np
from .neurite_object import NeuriteObject
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
    
    
class Neurite(NeuriteObject):
    """Represent one section of a rooted neurite tree."""

    def __init__(self, points=None, section_type=None, parent=None):
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
        super().__init__(section_type=section_type, parent=parent)

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
        """Return the length of the neurite."""
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

        The bins match the intervals used by ``sholl_plot``. Therefore, if the
        Sholl plot has length ``n``, each returned array has length ``n - 1``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(bifurcations, annihilations, internal_bifurcations)``.
        """


        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        # Standard neuron: the root is generally the soma and its children
        # are the primary dendrites.
        # discard soma, unknown, or sections with one point only
        neurites = {neurite for neurite in self.subtree if neurite.section_type not in {"soma", "unknown"} and len(neurite.points) > 1}

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


        # go over all neurites
        for neurite in neurites:
            i_bin = int(np.linalg.norm(neurite.points[-1] - source) / bin_size)
            match len(neurite.children):
                case 0:
                    annihilations[i_bin] += 1
                case 2:
                    same_section_type_cnt = sum(neurite.section_type == ch.section_type for ch in neurite.children)

                    match same_section_type_cnt:
                        case 1:
                            internal_bifurcations[i_bin] += 1
                        case 2:
                            bifurcations[i_bin] += 1
                        case _:
                            raise ValueError("A neurite have both children of different types.")
                case 1:
                    pass
                case _:
                    raise ValueError("A neurite have more then two children.")
        return bifurcations, annihilations, internal_bifurcations


    def _calculate_max_distance(self, bin_size):
        # position from which calculate distance
        source = self.root.points[0]

        distances = []
        for neurite in self.subtree:
            if neurite.section_type != "soma":
                for point in neurite.points:
                    distances.append(
                        np.linalg.norm(point - source)
                        )
        return max(distances)
    

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
        # discard soma, unknown, or sections with one point only
        neurites = {neurite for neurite in self.subtree if neurite.section_type not in {"soma", "unknown"} and len(neurite.points) > 1}

        # position from which calculate distance
        source = self.root.points[0]

        # create the vector counters
        if max_distance is None:
            max_distance = self._calculate_max_distance(bin_size)

        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        
        radii = np.arange(0.0, max_distance + bin_size, bin_size)
        crossings = np.zeros(len(radii), dtype=int)

        # get all the segments
        for neurite in neurites:
            for p0, p1 in zip(neurite.points[:-1], neurite.points[1:]):
                # distances
                start = np.linalg.norm(p0 - source)
                end = np.linalg.norm(p1 - source)
                
                # do not count any backward oriented segment
                if start < end:
                    crossings += (radii >= start) & (radii < end)
                    
        return crossings.astype(int)







