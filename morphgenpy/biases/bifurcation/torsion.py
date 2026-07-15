import numpy as np
from ..bias import BiasRegistry
from ... import misc
from ..elongation import all_dendrites_repulsion


@BiasRegistry.register_bifurcation("radial_torsion")
def radial_torsion_bias(neurite, angle):
    reference = neurite.last_direction

    repulsion = all_dendrites_repulsion(neurite, reference, None, None)

    if repulsion is not None:
        repulsion = misc._normalize(repulsion)
        
        last_direction = neurite.last_direction

        tmp = np.cross(last_direction, repulsion)

        if np.isfinite(tmp).all() and not np.isclose(np.linalg.norm(tmp), 0):
            lateral_direction = misc._normalize(tmp)
            return misc._normalize(last_direction - np.tan(angle) * lateral_direction), \
                   misc._normalize(last_direction + np.tan(angle) * lateral_direction)
            
##    return misc.AxialFrame.to_global(misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, np.pi]), neurite.last_direction), \
##           misc.AxialFrame.to_global(misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, 0.]), neurite.last_direction), \
               
    

    return misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, np.pi]), misc.EllipsoidalCoordinates.to_cartesian([1.0, angle, 0])
