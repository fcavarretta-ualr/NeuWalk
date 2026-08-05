import warnings
import numpy as np
from .section_synthesizer import SectionSynthesizer


def _get_candidates(targets, density, bin_size):
    """Return internal target steps weighted by their spatial density."""
    candidates = []

    for target in targets:
        for position in range(1, target.step_count - 1):
            distance = target.distance_from_root + position * target.step_size
            ibin = int(distance / bin_size)

            if 0 <= ibin < len(density) and density[ibin] > 0:
                candidates.append((target, position, density[ibin]))

    return candidates


def _tweak_section(section, position):
    """Split a section at an internal step and preserve its connectivity."""
    section_cont = SectionSynthesizer(section.step_size,
                                  label=section.label)
    # save step count
    tot_step_count = section.step_count

    # Update the continuation before changing the original section.
    section.step_count = position
    section_cont.step_count = tot_step_count - position

    # Transfer the original children to the continuation.
    for child in section.children:
        child.disconnect_from_parent()
        child.connect(section_cont, relation="parent")
        
    section_cont.connect(section, relation="parent")
    
    return section, section_cont


def _validate_internal_branches(branches, targets, rng, density, bin_size):
    """Validate inputs used to connect internal branches."""
    if not callable(getattr(rng, "random", None)):
        raise TypeError("rng must provide a random() method.")

    density = np.asarray(density, dtype=float)

    if density.ndim != 1 or density.size == 0:
        raise ValueError("density must be a nonempty 1D array.")
    if not np.all(np.isfinite(density)):
        raise ValueError("density must contain only finite values.")
    if np.any(density < 0):
        raise ValueError("density cannot contain negative values.")
    if not np.any(density > 0):
        raise ValueError("density must contain at least one positive value.")
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size must be a positive finite value.")

    for target in targets:
        if target.step_count < 3:
            continue
        if target.step_size <= 0:
            raise ValueError("Every target must have a positive step_size.")
        if not np.isfinite(target.distance_from_root):
            raise ValueError("Every target must have a finite distance_from_root.")


def connect_internal_branches(branches, targets, rng, density, bin_size):
    """Connect branches and return (branch, target_section) pairs."""
    _validate_internal_branches(branches, targets, rng, density, bin_size)

    branches = rng.permute(branches)
    targets = rng.permute(targets)
    connections = []

    while branches:
        candidates = _get_candidates(targets, density, bin_size)

        if not candidates:
            warnings.warn(
                f"No valid internal target steps remain; {len(branches)} branches were not connected.",
                RuntimeWarning,
            )
            break

        branch = branches.pop()
        target, position, _ = rng.choice(candidates, weights=[weight for _, _, weight in candidates])
        target, target_cont = _tweak_section(target, position)

        targets.append(target_cont)
        branch.connect(target, relation="parent")
        connections.append((branch, target))

        
    # increase the order of the obliques
    for branch, _ in connections:
        branch.set_order(branch.parent.order + 1)

    return connections
