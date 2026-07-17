import numpy as np
from morphgenpy.profiles.neurite import NeuriteProfile

def cut_neurite(rng, neurite, interval): 
    children = neurite.children.copy()
    neurite.disconnect_from_children()
    distal_neurite = neurite.clone()

    # min index
    min_index = int((interval[0] - neurite.distance_from_root) / neurite.step_size)
    max_index = int((interval[1] - neurite.distance_from_root) / neurite.step_size)
    
    # define the size of the two neurites
    new_step_count = int(rng.random() * (max_index - min_index + 1) + min_index)
    
    if new_step_count == neurite.step_count:
      print('distance', distance, neurite.distance_from_root, neurite.step_count, new_step_count)
      
    neurite.step_count = new_step_count
    distal_neurite.step_count -= new_step_count
    
    print(neurite.step_count, distal_neurite.step_count)
    distal_neurite.connect(neurite, relation="parent")

    for ch in children:
        ch.connect(distal_neurite, relation="parent")








def interval_overlap(a1, a2, b1, b2):
    """Return whether two intervals overlap and the overlapping interval."""
    start = max(a1, b1)
    end = min(a2, b2)
    return (start, end) if start < end else None

  
def neurites_by_distance_bin(roots, interval_start, interval_end):
    """
    Return neurites overlapping each distance interval.

    Each interval is interpreted as [start, end).
    """
    neurites = [ ]
    
    for root in roots:
      for neurite in root._iter_sections():
        if neurite.internal_bifurcation:
          continue
        # min and max cut position
        min_cut_position = neurite.distance_from_root + neurite.step_size
        max_cut_position = neurite.distance_from_root + neurite.length - neurite.step_size

        # check for the overlap with intervals
        match_interval = interval_overlap(min_cut_position, max_cut_position, interval_start, interval_end)
        #print('check 1', neurite.distance_from_root, neurite.length, neurite.step_count, neurite.section_type, neurite.parent.section_type)
        #print('\t', min_cut_position, max_cut_position, interval_start, interval_end, match_interval, neurite, neurite.section_type)
        if match_interval:
          neurites.append({
            'neurite':neurite,
            'interval':match_interval
            })

    return neurites
  
def choice(rng, size, probabilities):
    """
    Distribute exactly `size` elements across bins.

    Returns
    -------
    numpy.ndarray
        Number of elements assigned to each bin. The returned counts always
        sum to `size`.
    """
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("size must be an integer.")

    if size < 0:
        raise ValueError("size cannot be negative.")

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a nonempty 1D array.")

    if np.any(~np.isfinite(probabilities)):
        raise ValueError("probabilities must contain finite values.")

    if np.any(probabilities < 0.0):
        raise ValueError("probabilities cannot be negative.")

    total = probabilities.sum()

    if total <= 0.0:
        raise ValueError("At least one probability must be positive.")

    probabilities = probabilities / total
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0

    counts = np.zeros(len(probabilities), dtype=int)

    for _ in range(size):
        value = float(np.asarray(rng.random()).reshape(-1)[0])
        bin_index = np.searchsorted(cumulative, value, side="right")
        counts[bin_index] += 1

    return [
        (int(count), bin_index)
        for bin_index, count in enumerate(counts) if count > 0
    ]

def assign_neurites_to_bins(
    rng,
    n_neurites,
    internal_branching_density,
):
    """
    Assign each neurite to a bin according to internal-branching density.

    Returns
    -------
    numpy.ndarray
        Bin index assigned to each neurite. Entry i is the bin assigned to
        neurite i.
    """
    if not isinstance(n_neurites, int) or isinstance(n_neurites, bool):
        raise TypeError("n_neurites must be an integer.")

    if n_neurites < 0:
        raise ValueError("n_neurites cannot be negative.")

    density = np.asarray(
        internal_branching_density,
        dtype=float,
    )

    if density.ndim != 1 or density.size == 0:
        raise ValueError(
            "internal_branching_density must be a nonempty 1D array."
        )

    if not np.all(np.isfinite(density)):
        raise ValueError(
            "internal_branching_density must contain finite values."
        )

    if np.any(density < 0.0):
        raise ValueError(
            "internal_branching_density cannot contain negative values."
        )

    if n_neurites == 0:
        return np.empty(0, dtype=int)

    total_density = density.sum()

    if np.isclose(total_density, 0.0):
        raise ValueError(
            "internal_branching_density must contain at least one "
            "positive value."
        )

    probability = density / total_density

    return choice(
        rng,
        n_neurites,
        probability,
    )
