import numpy as np
    

class NeuriteProfile:
    """
    Represent a soma or neurite section during synthesis.

    A soma has zero length and path distance, cannot have a parent, and cannot
    elongate, bifurcate, or annihilate. It can only create primary dendrites.

    Other sections begin with one synthesis step and may elongate, bifurcate,
    bifurcate internally, or annihilate while active.
    """

    def __init__(self, step_size, section_type=None):
        """
        Initialize a soma or neurite section.

        Parameters
        ----------
        step_size : float
            Length represented by one synthesis step.
        section_type : optional
            Section label. Use ``"soma"`` for the soma.
        """
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.step_size = float(step_size)
        self.step_count = 0 if section_type == "soma" else 1
        self.order = 0
        self.children = []
        self._parent = None
        self.section_type = section_type
        self.active = True
        self.internal_bifurcation = False

    @property
    def parent(self):
        """Return the parent section."""
        return self._parent

    @parent.setter
    def parent(self, value):
        """Assign a parent to a non-soma section."""
        if self.section_type == "soma" and value is not None:
            raise RuntimeError("A soma cannot have a parent.")

        self._parent = value

    @property
    def length(self):
        """Return the section length."""
        if self.section_type == "soma":
            return 0.0

        return self.step_count * self.step_size

    @property
    def distance_from_root(self):
        """Return the path distance to the start of the section."""
        if self.section_type == "soma" or self.parent is None or self.parent.internal_bifurcation:
            return 0.0

        return self.parent.distance_from_root + self.parent.length

    @property
    def distance_from_soma(self):
        """Return the path distance to the start of the section."""
        return self.distance_from_root

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
            raise ValueError("number must be positive.")

        if section_type == "soma":
            raise ValueError(
                "A primary dendrite cannot have section_type='soma'."
            )

        self.children = [
            NeuriteProfile(
                step_size=self.step_size,
                section_type=section_type,
            )
            for _ in range(number)
        ]

        for child in self.children:
            child.parent = self

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

        self.children = self._create_children()
        self.internal_bifurcation = False
        self.active = False

        return self.children

    def undo_bifurcate(self):
        """Undo a terminal bifurcation."""
        if self.active:
            raise RuntimeError("The section has not bifurcated.")

        if self.internal_bifurcation:
            raise RuntimeError(
                "Use undo_bifurcate_internal() instead."
            )

        if len(self.children) != 2:
            raise RuntimeError("No bifurcation is available to undo.")

        self.children = []
        self.active = True

    def bifurcate_internal(self):
        """
        Create an internal bifurcation.

        The first child remains active and the second child is inactive.
        """
        self._check_event_allowed()

        self.children = self._create_children()
        self.children[1].active = False

        self.internal_bifurcation = True
        self.active = False

        return self.children

    def undo_bifurcate_internal(self):
        """Undo an internal bifurcation."""
        if self.active:
            raise RuntimeError(
                "The section has not bifurcated internally."
            )

        if not self.internal_bifurcation:
            raise RuntimeError(
                "No internal bifurcation is available to undo."
            )

        if len(self.children) != 2:
            raise RuntimeError(
                "No internal bifurcation is available to undo."
            )

        self.children = []
        self.internal_bifurcation = False
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

    def sholl_plot(self, bin_size=50, max_distance=None):
        """
        Calculate section crossings at increasing path distances.

        The soma is excluded from the calculation.
        """
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        if self.section_type == "soma":
            sections = [
                section
                for child in self.children
                for section in child._iter_sections()
            ]
        else:
            sections = list(self._iter_sections())

        if max_distance is None:
            max_distance = max(
                (
                    section.distance_from_root + section.length
                    for section in sections
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

        for section in sections:
            start = section.distance_from_root
            end = start + section.length

            crossings += (radii >= start) & (radii < end)

        return crossings

    def bifurcation_count(self):
        """
        Return the number of terminal bifurcations in the subtree.

        Soma primary dendrites and internal bifurcations are excluded.
        """
        count = int(
            self.section_type != "soma"
            and len(self.children) == 2
            and not self.internal_bifurcation
        )

        for child in self.children:
            count += child.bifurcation_count()

        return count

    def _create_children(self):
        """Create two child sections."""
        children = [
            NeuriteProfile(
                step_size=self.step_size,
                section_type=self.section_type,
            )
            for _ in range(2)
        ]

        for child in children:
            child.parent = self
            child.order = self.order + 1

        return children

    def _iter_sections(self):
        """Iterate over this section and all descendants."""
        yield self

        for child in self.children:
            yield from child._iter_sections()

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
        
    def has_children(self):
        """Return whether this neurite has child branches. """
        return len(self.children) > 0

if __name__ == '__main__':
  s = NeuriteProfile(step_size=1.0, section_type="soma")
  s.create_primary_dendrites(3, "apical_dendrite")

  children = []
  for ch in s.children:
      for _ in range(50):
        ch.elongate()

      children += ch.bifurcate()

      
  for ch in children:
    for _ in range(50):
      ch.elongate()

  print(s.sholl_plot())
