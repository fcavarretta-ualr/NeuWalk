import numpy as np
from ..bias import BiasRegistry
from ..... import misc

def plane_boundary(reference_section, origin, axis, K, n, strict=False):

  if (K is None) != (n is None):
    raise ValueError("K and n must both be provided or both be None.")

  axis_direction = misc.EllipsoidalCoordinates.to_cartesian((1., ) + axis)

  direction = reference_section.points[-1] - origin

  distance = np.dot(direction, axis_direction)

  if strict and distance < 0:
    return None

  factor = 1.0 if K is None else misc.hill(max(distance, 0), K, n)

  return axis_direction * factor



@BiasRegistry.register_elongation("plane_boundary")
def plane_boundary_bias(rng, reference_section, reference_direction, origin, axis, K, n, strict=False):
  return plane_boundary(reference_section, origin, axis, K, n, strict=strict)
