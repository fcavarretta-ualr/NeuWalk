from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


@dataclass(frozen=True)
class Preset:
    """Named morphology-generation preset."""

    name: str
    generate_fn: Callable[..., Any]
    fit_fn: Callable[..., Any] | None = None
    description: str = ""

    def generate(self, **kwargs):
        """Generate one morphology using this preset."""
        return self.generate_fn(**kwargs)

    def fit(self, *args, **kwargs):
        """Fit and save this preset from reconstructed morphologies."""
        if self.fit_fn is None:
            raise NotImplementedError(f"Preset {self.name!r} cannot be fitted.")
        return self.fit_fn(*args, **kwargs)
