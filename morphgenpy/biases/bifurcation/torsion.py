import numpy as np
from ..bias import BiasRegistry
from ... import misc
from .. import elongation

def _torsion_bias(rng, neurite, angle, repulsion):
    last_direction = neurite.last_direction

    tmp = np.cross(last_direction, repulsion)

    if np.isfinite(tmp).all() and not np.isclose(np.linalg.norm(tmp), 0):
      lateral_direction = misc.vector_normalize(tmp)
      return misc.vector_normalize(last_direction - np.tan(angle) * lateral_direction), \
             misc.vector_normalize(last_direction + np.tan(angle) * lateral_direction)    

    return None

@BiasRegistry.register_bifurcation("radial_torsion")
def radial_torsion_bias(rng, neurite, angle):
    repulsion = elongation.all_dendrites_repulsion(rng, neurite, neurite.last_direction, None, None)

    if repulsion is not None:
      directions = _torsion_bias(rng, neurite, angle, misc.vector_normalize(repulsion))
      if directions:
        return directions

    return misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, np.pi]), misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, 0])
 



@BiasRegistry.register_bifurcation("internal_branch")
def internal_branch_bias(rng, neurite, angle):
    return neurite.last_direction, \
           misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, rng.random() * 2 * np.pi])






@BiasRegistry.register_bifurcation("cross_torsion")
def cross_torsion_bias(rng, neurite, angle, **kwargs):

  match kwargs.get('space'):
    case 'ellipsoid':
      repulsion = misc.EllipsoidalCoordinates.normal_direction(neurite.points[-1], kwargs.get('radii'), center=kwargs.get('center'))
    case 'space':
      repulsion = misc.EllipsoidalCoordinates.normal_direction(neurite.points[-1], np.ones(3), center=kwargs.get('center'))
    case _:
      raise ValueError("Unknown space")

  return _torsion_bias(rng, neurite, angle, repulsion)
      
    
