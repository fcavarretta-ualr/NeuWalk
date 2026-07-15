import numpy as np
from ..bias import BiasRegistry
from ... import misc

def truncated_cone_boundary(
    point,
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
    rel_point = misc.AxialFrame.to_local(point, axis_direction, center=origin)

    # calculate the radiuses
    
    t = rel_point[2] / length

    # with strict flag do not return any bias if outside the boundaries
    if strict and t < 0 or t > 1:
        return None
    
    if t < 0:
        t = 0.
    rel_radii = [
        radii_1[0] + (radii_2[0] - radii_1[0]) * t,
        radii_1[1] + (radii_2[1] - radii_1[1]) * t
        ]


    # project the point on a concentrical ellipse
    coordinates = misc.EllipsoidalCoordinates.from_cartesian(rel_point[:-1], rel_radii)



    # relative coordinates
    rel_extreme_1 = np.append(
        misc.EllipsoidalCoordinates.to_cartesian(coordinates, radii_1),
        0.)
    
    rel_extreme_2 = np.append(
        misc.EllipsoidalCoordinates.to_cartesian(coordinates, radii_2),
        length)


    # get the two extreme to calculate the bias direction
    extreme_1 = misc.AxialFrame.to_global(rel_extreme_1, axis_direction, center=origin)
    extreme_2 = misc.AxialFrame.to_global(rel_extreme_2, axis_direction, center=origin)   

    
    bias_direction = misc._normalize(extreme_2 - extreme_1)
    
    factor = misc.hill(coordinates[0], K, n)
    return bias_direction, factor



##def truncated_cone_boundary(
##    point,
##    origin,
##    axis,
##    radii_1,
##    radii_2,
##    K,
##    n,
##):
##    if np.isscalar(radii_1):
##        radii_1 = (radii_1, radii_1, radii_1)
##
##    if np.isscalar(radii_2):
##        radii_2 = (radii_2, radii_2, radii_2)
##
##    length, theta, phi = axis
##
##    # axis alignment
##    axis_direction = np.array([
##        np.sin(theta) * np.cos(phi),
##        np.sin(theta) * np.sin(phi),
##        np.cos(theta),
##    ])
##
##    # relative coordinates of the point
##    rel_point = misc.AxialFrame.to_local(point, axis_direction, center=origin)
##
##    # calculate the radiuses
##    t = np.clip(rel_point[2] / length, 0.0, 1.0)
##    rel_radii = [
##        radii_1[0] + (radii_2[0] - radii_1[0]) * t,
##        radii_1[1] + (radii_2[1] - radii_1[1]) * t
##        ]
##
##
##    # project the point on a concentrical ellipse
##    radial_distance, _ = misc.EllipsoidalCoordinates.from_cartesian(rel_point[:-1], rel_radii)
##    
##    #radial_distance = np.sqrt((x / a) ** 2 + (y / b) ** 2)
##    #factor = misc.hill(radial_distance, K, n)
##    factor = misc.hill(radial_distance, K, n)
##    return axis_direction, factor


@BiasRegistry.register_elongation("truncated_cone_boundary")
def truncated_cone_boundary_bias(
    reference_dendrite,
    reference_direction,
    origin,
    axis,
    radii_1,
    radii_2,
    K,
    n
):
    # ghost point
    ghost_point = reference_dendrite._generate_point(reference_direction)

    output0 = truncated_cone_boundary(
        reference_dendrite.current_point,
        origin,
        axis,
        radii_1,
        radii_2,
        K,
        n
    )

    output1 = truncated_cone_boundary(
        ghost_point,
        origin,
        axis,
        radii_1,
        radii_2,
        K,
        n
    )

    if not (output0 and output1):
        return None
    
    axis_direction0, factor0 = output0
    axis_direction1, factor1 = output1
    
    # result direction
    direction = axis_direction0 * factor0 + axis_direction1 * factor1
    
    if np.isclose(np.linalg.norm(direction), 0.):
        return None
    
    return misc._normalize(direction) * (factor0 + factor1) * 0.5
