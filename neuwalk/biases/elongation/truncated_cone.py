import numpy as np
from ..bias import BiasRegistry
from ... import misc

def truncated_cone_boundary(
    reference_dendrite,
    reference_direction, 
    origin,
    axis,
    radii_1,
    radii_2,
    K,
    n,
    strict=False
):
    if np.isscalar(radii_1):
        radii_1 = (radii_1, radii_1, radii_1)

    if np.isscalar(radii_2):
        radii_2 = (radii_2, radii_2, radii_2)

    length, theta, phi = axis

    # axis alignment
    axis_direction = misc.EllipsoidalCoordinates.to_cartesian((1.0, theta, phi))

    # relative coordinates of the point
    rel_point = misc.AxialFrame.to_local(reference_dendrite.current_point, axis_direction, center=origin)

    # calculate the radiuses
    t = rel_point[2] / length

    # with strict flag do not return any bias if outside the boundaries
    if strict and (t < 0 or t > 1):
        return None
    
    # dest point
    dest_point = t * axis_direction * length + origin

    # direction
    direction = dest_point - reference_dendrite.current_point

    if np.isclose(np.linalg.norm(direction), 0.0):
        return None
    
    if np.dot(reference_direction, direction) > 0:         
        return np.zeros(3)
    
    if t < 0:

        factor = 1

    else:
        
        rel_radii = [
            radii_1[0] + (radii_2[0] - radii_1[0]) * t,
            radii_1[1] + (radii_2[1] - radii_1[1]) * t
            ]


        rho, phi = misc.EllipsoidalCoordinates.from_cartesian(rel_point[:-1], rel_radii)

        # distance dependent factor
        factor = misc.hill(rho, K, n)
    

    #print(direction, reference_dendrite.current_point, dest_point)
    return misc.vector_normalize(direction) * factor









@BiasRegistry.register_elongation("truncated_cone_boundary")
def truncated_cone_boundary_bias(
    rng,
    reference_dendrite,
    reference_direction, 
    origin,
    axis,
    radii_1,
    radii_2,
    K,
    n,
    strict=False
):


    return truncated_cone_boundary(
        reference_dendrite,
        reference_direction, 
        origin,
        axis,
        radii_1,
        radii_2,
        K,
        n,
        strict=strict
    )






