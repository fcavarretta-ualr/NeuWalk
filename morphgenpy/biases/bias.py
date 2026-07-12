import numpy as np


class Bias:
    """Callable bias object supporting linear combinations."""

    def __init__(self, compute=None, terms=None):
        self._compute = compute
        self._terms = terms

    def compute(self, random_walk, point):
        if self._terms is None:
            return self._compute(random_walk, point)

        values = []

        for weight, bias in self._terms:
            value = bias.compute(random_walk, point)

            if value is not None:
                values.append(
                    weight * np.asarray(value, dtype=float)
                )

        return None if not values else np.sum(values, axis=0)

    def __add__(self, other):
        if not isinstance(other, Bias):
            return NotImplemented

        return Bias(terms=[
            (1.0, self),
            (1.0, other),
        ])

    def __mul__(self, weight):
        if not np.isscalar(weight):
            return NotImplemented

        return Bias(terms=[
            (float(weight), self),
        ])

    __rmul__ = __mul__



class BiasRegistry:
    """Register and retrieve bias functions by identifier."""

    _functions = {}

    @classmethod
    def register(cls, identifier, function=None):
        """Register a bias function, optionally as a decorator."""
        def decorator(func):
            if identifier in cls._functions:
                raise KeyError(
                    f"Bias {identifier!r} is already registered."
                )

            cls._functions[identifier] = func
            return func

        return decorator(function) if function is not None else decorator

    @classmethod
    def get(cls, identifier, *args, **kwargs):
        """Return a bias function with additional parameters bound."""
        try:
            function = cls._functions[identifier]
        except KeyError:
            raise KeyError(
                f"Unknown bias identifier: {identifier!r}."
            ) from None

        def bias_function(random_walk, point):
            return function(
                random_walk,
                point,
                *args,
                **kwargs,
            )

        return bias_function

    @classmethod
    def available(cls):
        """Return the registered identifiers."""
        return tuple(cls._functions)
