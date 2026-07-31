import importlib
import pkgutil
from .. import misc
import numpy as np


class ElongationBias:
    """Elongation bias supporting weighted linear combinations."""

    def __init__(self, compute=None, terms=None):
        if compute is None and terms is None:
            raise ValueError("Either compute or terms must be provided.")

        self._compute = compute
        self._terms = terms

    def compute(self, rng, random_walk, reference_direction):
        """Compute the elongation bias vector."""
        if self._terms is None:
            return self._compute(rng, random_walk, reference_direction)

        values = []

        for weight, bias in self._terms:
            value = bias.compute(rng, random_walk, reference_direction)

            if value is not None:
                values.append(weight * np.asarray(value, dtype=float))

        return None if not values else np.sum(values, axis=0)

    def __add__(self, other):
        if not isinstance(other, ElongationBias):
            return NotImplemented

        return ElongationBias(
            terms=[
                (1.0, self),
                (1.0, other),
            ]
        )

    def __mul__(self, weight):
        if not np.isscalar(weight):
            return NotImplemented

        return ElongationBias(
            terms=[
                (float(weight), self),
            ]
        )

    __rmul__ = __mul__


class BifurcationBias:
    """Simple bifurcation bias wrapper."""

    def __init__(self, compute):
        if not callable(compute):
            raise TypeError("compute must be callable.")

        self._compute = compute

    def compute(self, rng, section):
        """Compute the bifurcation directions for a section."""
        return self._compute(rng, section)


class BiasRegistry:
    """Register and retrieve elongation and bifurcation biases."""

    _elongation = {}
    _bifurcation = {}
    _discovered = False

    @classmethod
    def register_elongation(cls, identifier, function=None):
        """Register an elongation bias function."""
        return cls._register(
            cls._elongation,
            "elongation",
            identifier,
            function,
        )

    @classmethod
    def register_bifurcation(cls, identifier, function=None):
        """Register a bifurcation bias function."""
        return cls._register(
            cls._bifurcation,
            "bifurcation",
            identifier,
            function,
        )

    @staticmethod
    def _register(registry, kind, identifier, function=None):
        """Register a function, optionally as a decorator."""
        def decorator(func):
            if identifier in registry:
                raise KeyError(
                    f"{kind.capitalize()} bias "
                    f"{identifier!r} is already registered."
                )

            registry[identifier] = func
            return func

        return decorator(function) if function is not None else decorator

    @classmethod
    def get_elongation(cls, identifier, *args, **kwargs):
        """Return a configured ElongationBias."""
        cls.discover()

        try:
            function = cls._elongation[identifier]
        except KeyError:
            raise KeyError(
                f"Unknown elongation bias: {identifier!r}."
            ) from None

        def compute(rng, random_walk, reference_direction):
            return function(
                rng,
                random_walk,
                reference_direction,
                *args,
                **kwargs,
            )

        return ElongationBias(compute=compute)

    @classmethod
    def get_bifurcation(cls, identifier, *args, **kwargs):
        """Return a configured BifurcationBias."""
        cls.discover()

        try:
            function = cls._bifurcation[identifier]
        except KeyError:
            raise KeyError(
                f"Unknown bifurcation bias: {identifier!r}."
            ) from None

        def compute(rng, section):
            return function(rng, section, *args, **kwargs)

        return BifurcationBias(compute)

    @classmethod
    def available_elongation(cls):
        """Return registered elongation bias identifiers."""
        cls.discover()
        return tuple(cls._elongation)

    @classmethod
    def available_bifurcation(cls):
        """Return registered bifurcation bias identifiers."""
        cls.discover()
        return tuple(cls._bifurcation)

    @classmethod
    def available(cls):
        """Return all registered identifiers."""
        cls.discover()

        return (
            *(
                f"elongation:{name}"
                for name in cls._elongation
            ),
            *(
                f"bifurcation:{name}"
                for name in cls._bifurcation
            ),
        )

    @classmethod
    def discover(cls):
        """Import modules from the elongation and bifurcation packages."""
        if cls._discovered:
            return

        cls._discovered = True

        for category in ("elongation", "bifurcation"):
            package = importlib.import_module(
                f"{__package__}.{category}"
            )

            for module in pkgutil.iter_modules(
                package.__path__,
                prefix=f"{package.__name__}.",
            ):
                name = module.name.rsplit(".", 1)[-1]

                if not name.startswith("_"):
                    importlib.import_module(module.name)
