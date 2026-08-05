import numpy as np
from ..bias import BiasRegistry
from ..... import misc
from .. import elongation

def _torsion_bias(rng, section, angle, repulsion):
    last_direction = section.last_direction

    tmp = np.cross(last_direction, repulsion)

    if np.isfinite(tmp).all() and not np.isclose(np.linalg.norm(tmp), 0):
      lateral_direction = misc.to_unit_vector(tmp)
      return misc.to_unit_vector(last_direction - np.tan(angle) * lateral_direction), \
             misc.to_unit_vector(last_direction + np.tan(angle) * lateral_direction)    

    return None

@BiasRegistry.register_bifurcation("radial_torsion")
def radial_torsion_bias(rng, section, angle):
    repulsion = elongation.all_sections_repulsion(rng, section, section.last_direction, None, None)

    if repulsion is not None:
      directions = _torsion_bias(rng, section, angle, misc.to_unit_vector(repulsion))
      if directions:
        return directions

    return misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, np.pi]), misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, 0])
 



@BiasRegistry.register_bifurcation("internal_branch")
def internal_branch_bias(rng, section, angle):
    return section.last_direction, \
           misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, rng.random() * 2 * np.pi])






@BiasRegistry.register_bifurcation("cross_torsion")
def cross_torsion_bias(rng, section, angle, **kwargs):

  match kwargs.get('space'):
    case 'ellipsoid':
      repulsion = misc.EllipsoidalCoordinates.normal_direction(section.points[-1], kwargs.get('radii'), center=kwargs.get('center'))
    case 'space':
      repulsion = misc.EllipsoidalCoordinates.normal_direction(section.points[-1], np.ones(3), center=kwargs.get('center'))
    case _:
      raise ValueError("Unknown space")

  return _torsion_bias(rng, section, angle, repulsion)
      
    
