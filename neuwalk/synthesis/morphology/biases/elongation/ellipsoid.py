import numpy as np
from ..bias import BiasRegistry
from ..... import misc

def ellipsoid_boundary(
  reference_section,
  reference_direction,
  radii,
  K,
  n,
  depth=None,
  center=None,
  orientation=None,
  strict=False,
  resistance=True):

  

  if center is None:
    center = np.zeros(3)
      
  if center.shape != (3,):
    raise ValueError("center must be a 3D vector with shape (3,).")


  if np.any(radii <= 0):
    raise ValueError("All ellipsoid radii must be positive.")

  if orientation is None:
    orientation = 'in'

  if orientation not in {"in", "out"}:
      raise ValueError("orientation must be either 'in' or 'out'.")
    
  if (K is None) != (n is None):
      raise ValueError("K and n must both be provided or both be None.")



      
  # this make it inward
  sign = -1 if orientation == "in" else 1
  
  # normal distance
  distance = sign * misc.EllipsoidalCoordinates.depth(reference_section.points[-1], radii, center=center)

  if strict:
    current_depth = sign * distance

    # if below (above) the surface and deeper than indicated by depth
    if current_depth < 0 or (depth and current_depth > depth):
      return None
    

  # the orientation is outward by default
  normal_direction = sign * misc.EllipsoidalCoordinates.normal_direction(reference_section.points[-1], radii, center=center)

  # if it is already aligned, do not correct
  if resistance and np.dot(normal_direction, reference_direction) > 0:
    return None


  
  factor = 1 if K is None else misc.hill(distance, K, n) 

  return normal_direction * factor                         

  
@BiasRegistry.register_elongation("ellipsoid_boundary")
def ellipsoid_boundary_bias(
    rng,
    reference_section,
    reference_direction,
    radii,
    K,
    n,
    depth=None,
    center=None,
    orientation=None,
    strict=False,
    resistance=True
):


    return ellipsoid_boundary(
      reference_section,
      reference_direction,
      radii,
      K,
      n,
      depth=depth,
      center=center,
      orientation=orientation,
      strict=strict,
      resistance=resistance
    )
