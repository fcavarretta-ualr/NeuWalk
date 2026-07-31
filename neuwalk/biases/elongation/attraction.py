import numpy as np
from ..bias import BiasRegistry

@BiasRegistry.register_elongation("attraction")
def attraction_bias(rng, reference_section, reference_direction, K, n, destination_point):
    
    direction = destination_point - reference_section.points[-1]
    
    distance = np.linalg.norm(destination_point - reference_section.points[-1])

    if np.isclose(distance, 0.):
      return None
    
    if (K is None) != (n is None):
        raise ValueError("K and n must both be provided or both be None.")
    
    factor = 1.0 if K is None else misc.hill(distance, K, n)

    return direction / distance * factor
   

