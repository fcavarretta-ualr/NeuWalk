import numpy as np
from ..core.neurite_object import NeuriteObject


class NeuriteProfile(NeuriteObject):
    """
    Represent a soma or neurite section during synthesis.

    A soma has zero length and path distance, cannot have a parent, and cannot
    elongate, bifurcate, or annihilate. It can only create primary dendrites.

    Other sections begin with one synthesis step and may elongate, bifurcate,
    bifurcate internally, or annihilate while active.
    """

    def __init__(self, step_size, section_type=None, parent=None):
        """
        Initialize a soma or neurite section.

        Parameters
        ----------
        step_size : float
            Length represented by one synthesis step.
        section_type : optional
            Section label. Use ``"soma"`` for the soma.
        """

        super().__init__(section_type=section_type, parent=parent)
        
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.step_size = float(step_size)
        self.step_count = 1 if section_type != "soma" else 0
        self.order = 0
        self.active = True

    @property
    def length(self):
        """Return the section length."""
        if self.section_type == "soma":
            return 0.0

        return self.step_count * self.step_size


    def create_primary_dendrites(self, number, section_type):
        """Create primary dendrites from the soma."""
        if self.section_type != "soma":
            raise RuntimeError(
                "Primary dendrites can only be created from a soma."
            )

        if self.children:
            raise RuntimeError(
                "Primary dendrites have already been created."
            )

        if not isinstance(number, int) or isinstance(number, bool):
            raise TypeError("number must be an integer.")

        if number <= 0:
            raise ValueError(f"number must be positive. {number}")

        if section_type == "soma":
            raise ValueError(
                "A primary dendrite cannot have section_type='soma'."
            )

        for _ in range(number):
            self.connect(
                NeuriteProfile(self.step_size, section_type=section_type),
                relation="child")

        return self.children

    def elongate(self):
        """Increase the section step count."""
        self._check_event_allowed()
        self.step_count += 1

    def undo_elongate(self):
        """Undo one elongation step."""
        if self.section_type == "soma":
            raise RuntimeError("A soma cannot undo elongation.")

        if self.step_count <= 1:
            raise RuntimeError("No elongation is available to undo.")

        self.step_count -= 1

    def bifurcate(self):
        """Create two active children and deactivate the parent."""
        self._check_event_allowed()

        for _ in range(2):
            self.connect(
                NeuriteProfile(self.step_size, section_type=self.section_type),
                relation="child"
                )
            
        self.active = False

        return self.children

    def undo_bifurcate(self):
        """Undo a terminal bifurcation."""
        if self.active:
            raise RuntimeError("The section has not bifurcated.")


        if len(self.children) != 2:
            raise RuntimeError("No bifurcation is available to undo.")

        self.disconnect_from_children()
        self.active = True

    def annihilate(self):
        """Deactivate the section."""
        self._check_event_allowed()
        self.active = False

    def undo_annihilate(self):
        """Undo annihilation and reactivate the section."""
        if self.section_type == "soma":
            raise RuntimeError("A soma cannot undo annihilation.")

        if self.active:
            raise RuntimeError("The section is already active.")

        if self.children:
            raise RuntimeError(
                "A section with children was not annihilated."
            )

        self.active = True
        
    def _check_event_allowed(self):
        """Check whether the section can perform a synthesis event."""
        if self.section_type == "soma":
            raise RuntimeError(
                "A soma cannot elongate, bifurcate, or annihilate."
            )

        if not self.active:
            raise RuntimeError(
                "An inactive neurite section cannot perform an event."
            )

    def _calculate_max_distance(self, bin_size):
        # position from which calculate distance
        return max(
            neurite.distance_from_root + neurite.length for neurite in self.subtree if neurite.section_type != "soma")
    

    def sholl_plot(self, bin_size=50, max_distance=None):
        """
        Calculate section crossings at increasing path distances.

        The soma is excluded from the calculation.
        """
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        neurites = {neurite for neurite in self.subtree if neurite.section_type not in {"soma", "unknown"} and neurite.length > 0}

        # create the vector counters
        if max_distance is None:
            max_distance = self._calculate_max_distance(bin_size)

        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        radii = np.arange(
            0.0,
            max_distance + bin_size,
            bin_size,
        )

        crossings = np.zeros(len(radii), dtype=int)

        for neurite in neurites:
            start = neurite.distance_from_root
            end = start + neurite.length

            crossings += (radii >= start) & (radii < end)

        return crossings

