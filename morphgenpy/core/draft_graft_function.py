import numpy as np


def assign_oblique(rng, bin_size, profiles, n_assignments, densities):
    """
    Randomly assign an oblique to a profile and an internal location.

    Returns
    -------
    profile : NeuriteProfile
        Profile receiving the oblique.
    step_index : int
        Local synthesis step within the profile.
    distance_from_soma : float
        Path distance of the oblique origin from the soma.
    """
    # estimate probabilites
    probabilities = np.asarray(densities, dtype=float)

    # counter of availabile segments
    ref = [ [] for _ in range(probabilities.size) ]

    union_find = {}
    
    # Collect every valid internal location and its spatial interval.
    for profile in profiles:
      for step in range(1, profile.step_count):
        distance = profile.distance_from_root + step * profile.step_size
        bin_index = int(distance / bin_size)
        ref[bin_index].append((profile, step))
        union_find[(profile, step)] = profile
        

    for _ in n_assignments:
      non_empty = np.array([ i for i in range(len(ref)) if ref[i]], dtype=int)
      cdf = np.cumsum(probabilities[non_empty])
      cdf /= cdf[-1]
      
      X, Y = rng.random(n=2)
      
      index = np.searchsorted(cdf, X, side="left")
      
      cut = np.searchsorted(ref[index], Y, side="left")

      # clone the neurite, cut, connect
      
      # cut and use union find
      
      # increase the order +1 with respect to parent
      
      # profiles are changed within the function, no return is needed.
