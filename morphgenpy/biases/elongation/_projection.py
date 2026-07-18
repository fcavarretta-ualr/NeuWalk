from morphgenpy import misc
import numpy as np

def _project(point, direction, shape, **kwargs):
    center = kwargs.get('center')
    
    if center is None:
      center = np.zeros(3)
      
    if center.shape != (3,):
        raise ValueError("center must be a 3D vector with shape (3,).")

    if shape == "ellipsoid":
      normal_direction = misc.EllipsoidalCoordinates.to_cartesian(point, kwargs.get('radii'), center=center)
    elif shape == "plane":
      normal_direction = misc.EllipsoidalCoordinates.to_cartesian((1.0, theta, phi), center=center)
    else:
      raise ValueError("space must be 'ellipsoid' or 'plane'.")
    
    

    e1 = misc.AxialFrame.to_global(np.array([1., 0., 0.]), normal_direction)
    e2 = misc.AxialFrame.to_global(np.array([0., 1., 0.]), normal_direction)

    return misc._normalize(np.dot(direction, e1) * e1 + np.dot(direction, e2) * e2)

def _project_ellipsoid(point, radii, direction, center=None):
  return _project(
    point, direction, 'ellipsoid', radii=radii, center=center
    )

    
def _project_plane(point, theta, phi, direction, center=None):
  return _project(
    point, direction, 'plane', theta=theta, phi=phi, center=center
    )

def project(neurite, direction, **kwargs):
    space = kwargs.get('space')
    match space:
      case "ellipsoid":
        projected_direction = _project_ellipsoid(neurite.points[-1], kwargs.get('radii'), direction, center=kwargs.get('center'))
      case "plane":
        projected_direction = _project_plane(neurite.points[-1], kwargs.get('theta', 0), kwargs.get('phi', 0), direction, center=kwargs.get('center'))
      case _:
        return direction

    return projected_direction * np.linalg.norm(direction)
