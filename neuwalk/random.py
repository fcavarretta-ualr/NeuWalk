import copy

import numpy as np


class Random:
    """Small wrapper around a random-number generator."""

    def __init__(self, seed=None):
        """
        Parameters
        ----------
        seed : int, optional
            Seed used to initialize the NumPy random-number generator.
        """
        self._rng = np.random.default_rng(seed)

    def random(self, n=None):
        """
        Generate random values in the interval [0, 1).

        Parameters
        ----------
        n : int, optional
            Number of values to generate. When omitted, return a scalar.

        Returns
        -------
        float or numpy.ndarray
            One random value or an array with shape ``(n,)``.
        """
        if n is None:
            return float(self._rng.random())

        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer or None.")

        if n < 0:
            raise ValueError("n cannot be negative.")

        return self._rng.random(n)

    def permute(self, vector):
        """
        Return a random permutation of ``vector``.

        Parameters
        ----------
        vector : sequence
            Sequence to permute. It is not modified in place.

        Returns
        -------
        list
            A new list containing every element of ``vector`` in a
            random order.
        """
        vector = list(vector)

        result = []

        while len(vector):
            i = int(self.random() * len(vector))
            result.append(vector.pop(i))

        return result

    def choice(self, vector, weights=None):
        """
        Randomly select one element from ``vector``.

        Parameters
        ----------
        vector : sequence
            Sequence to choose from.
        weights : array-like, optional
            Relative weight of each element in ``vector``, in the same
            order. When omitted, every element is equally likely.

        Returns
        -------
        object
            The selected element of ``vector``.
        """
        vector = list(vector)

        if not vector:
            raise ValueError("vector cannot be empty.")

        if weights is None:
            index = int(self.random() * len(vector))
            return vector[index]

        weights = np.asarray(weights, dtype=float)

        if weights.shape != (len(vector),):
            raise ValueError("weights must have the same length as vector.")

        if np.any(weights < 0):
            raise ValueError("weights cannot contain negative values.")

        total = weights.sum()

        if not np.isfinite(total) or total <= 0:
            raise ValueError("weights must contain at least one positive value.")

        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        index = int(np.searchsorted(cdf, self.random(), side="left"))

        return vector[index]

    def clone(self):
        """
        Return a new Random object with the same internal state.

        The clone produces the same future sequence as the original until
        either object is advanced independently.
        """
        clone = self.__class__()
        clone._rng.bit_generator.state = copy.deepcopy(
            self._rng.bit_generator.state
        )
        return clone
