import numpy as np
from ..bias import BiasRegistry
from ... import misc

def truncated_cone_boundary(
    reference_dendrite,
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
    axis_direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # relative coordinates of the point
    rel_point = misc.AxialFrame.to_local(reference_dendrite.current_point, axis_direction, center=origin)

    # calculate the radiuses
    t = rel_point[2] / length

    # with strict flag do not return any bias if outside the boundaries
    if strict and (t < 0 or t > 1):
        return None
        
    rel_radii = [
        radii_1[0] + (radii_2[0] - radii_1[0]) * t,
        radii_1[1] + (radii_2[1] - radii_1[1]) * t
        ]


    # project the point on a concentrical ellipse

    rho, phi = misc.EllipsoidalCoordinates.from_cartesian(rel_point[:-1], rel_radii)

    # distance dependent factor
    factor = misc.hill(rho, K, n)
    
    if rho <= 1:         
        return axis_direction * factor
        

    # relative coordinates
    rel_extreme = np.append(
        misc.EllipsoidalCoordinates.to_cartesian((rho, phi), rel_radii),
        rel_point[2])

    # get the two extreme to calculate the bias direction
    extreme = misc.AxialFrame.to_global(rel_extreme, axis_direction, center=origin)

    if np.isclose(np.linalg.norm(extreme - reference_dendrite.current_point), 0):
            return axis_direction * factor

    return misc._normalize(extreme - reference_dendrite.current_point) * factor








@BiasRegistry.register_elongation("truncated_cone_boundary")
def truncated_cone_boundary_bias(
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


    direction = truncated_cone_boundary(
        reference_dendrite,
        origin,
        axis,
        radii_1,
        radii_2,
        K,
        n,
        strict=strict
    )



    if direction is None:
        return None

    return direction





