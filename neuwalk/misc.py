import numpy as np

def translate_points(points, source, target=None):
    """
    Translate points so that ``source`` is moved to ``target``.

    Parameters
    ----------
    points : array-like
        Point or array of points with final dimension 3.
    source : array-like
        Point to translate from.
    target : array-like, optional
        Destination point. Defaults to the origin.

    Returns
    -------
    numpy.ndarray
        Translated copy of the points.
    """
    is_list = type(points) == list

    if target is None:
        target = np.zeros(3)
    else:
        target = target.copy()

    source = source.copy()
    
    return [ p.copy() + target - source for p in points.copy() ] 

def to_unit_vector(vector):
    """Return a normalized 3D vector."""
    vector = np.asarray(vector, dtype=float)

    if vector.shape != (3,):
        raise ValueError("The vector must have shape (3,).")

    norm = np.linalg.norm(vector)

    if np.isclose(norm, 0.0):
        raise ValueError(
            "A zero vector cannot define a direction."
        )

    return vector / norm





def hill(x, half, exponent=1.0, minimum=0.0, maximum=1.0):
    """
    Generalized Hill function for scalar or array inputs.

    Negative exponents produce a decreasing Hill function.
    """
    x = np.asarray(x, dtype=float)

    if half <= 0:
        raise ValueError("half must be positive.")

    x[x < 0] = 0.

    z = exponent * (np.log(x, where=x > 0, out=np.full_like(x, -np.inf))
                    - np.log(half))

    # Numerically stable logistic function.
    value = np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )

    result = minimum + (maximum - minimum) * value

    return result.item() if result.ndim == 0 else result


def sphere_surface_points(
    n,
    theta=(0.0, np.pi),
    phi=(0.0, 2.0 * np.pi),
    radius=1.0,
    center=None,
):
    """
    Generate deterministic points on a spherical surface region.

    ``theta`` is the polar angle measured from the positive z-axis and
    must lie in ``[0, pi]``. It is a bounded, non-periodic interval: 0
    and pi are the two distinct poles, not the same point.

    ``phi`` is the azimuthal angle in the x-y plane and is periodic over
    ``2*pi``: 0 and 2*pi refer to the same point.

    ``theta`` and ``phi`` may each be either:

    - a scalar, fixing that angle;
    - a two-element sequence defining an angular interval.

    When only one of ``theta``/``phi`` is an interval (the other fixed),
    the ``n`` points are equally spaced over that interval:

    - a ranging ``theta`` uses ``n`` points equally spaced over
      ``[theta_min, theta_max]``, including both endpoints (as for
      ``numpy.linspace``), since the two ends are genuinely distinct
      poles.
    - a ranging ``phi`` uses ``n`` points equally spaced starting at
      ``phi_start`` and stepping by ``phi_span / n``, without a point
      placed at the far end of the interval: for a full ``2*pi`` span
      that far end is the same point as ``phi_start``, so including it
      would duplicate a point.

    When both angles are intervals, the ``n`` points are distributed
    approximately uniformly, with respect to spherical surface area,
    over the patch of the sphere they bound: equally spaced by area in
    the polar direction (uniform in ``cos(theta)``), and spread across
    the azimuthal range with a golden-angle step so points don't line
    up into meridional rows. No random sampling is used anywhere in
    this function.

    Parameters
    ----------
    n : int
        Number of points.
    theta : float or sequence of two floats, optional
        Polar angle or polar-angle interval.
    phi : float or sequence of two floats, optional
        Azimuthal angle or azimuthal-angle interval.
        Intervals may wrap through ``2*pi``.
    radius : float, default 1
        Sphere radius.
    center : array_like, optional
        Sphere center. Default is ``[0, 0, 0]``.

    Returns
    -------
    numpy.ndarray
        Points with shape ``(n, 3)``.
    """
    if (
        not isinstance(n, int)
        or isinstance(n, bool)
    ):
        raise TypeError("n must be an integer.")

    if n < 0:
        raise ValueError("n cannot be negative.")

    radius = float(radius)

    if not np.isfinite(radius):
        raise ValueError("radius must be finite.")

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if center is None:
        center = np.zeros(3, dtype=float)
    else:
        center = np.asarray(center, dtype=float)

    if center.shape != (3,):
        raise ValueError(
            "center must have shape (3,)."
        )

    if not np.isfinite(center).all():
        raise ValueError(
            "center must contain finite values."
        )

    if n == 0:
        return np.empty((0, 3), dtype=float)

    def parse_angle(value, name):
        if np.isscalar(value):
            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            return value, value, True

        value = np.asarray(value, dtype=float)

        if value.shape != (2,):
            raise ValueError(
                f"{name} must be a scalar or "
                "a two-element sequence."
            )

        if not np.isfinite(value).all():
            raise ValueError(
                f"{name} must contain finite values."
            )

        return (
            float(value[0]),
            float(value[1]),
            False,
        )

    theta_min, theta_max, theta_scalar = (
        parse_angle(theta, "theta")
    )

    if not (
        0.0 <= theta_min <= np.pi
        and 0.0 <= theta_max <= np.pi
    ):
        raise ValueError(
            "theta values must lie in [0, pi]."
        )

    if theta_min > theta_max:
        raise ValueError(
            "theta minimum cannot exceed theta maximum."
        )

    theta_fixed = (
        theta_scalar
        or np.isclose(theta_min, theta_max)
    )

    phi_start, phi_end, phi_scalar = (
        parse_angle(phi, "phi")
    )

    two_pi = 2.0 * np.pi

    if phi_scalar:
        phi_start %= two_pi
        phi_span = 0.0
        phi_fixed = True

    else:
        raw_span = phi_end - phi_start

        if abs(raw_span) > two_pi and not np.isclose(
            abs(raw_span),
            two_pi,
        ):
            raise ValueError(
                "phi interval cannot span more than 2*pi."
            )

        phi_start %= two_pi

        if np.isclose(abs(raw_span), two_pi):
            phi_span = two_pi
            phi_fixed = False

        elif np.isclose(raw_span, 0.0):
            phi_span = 0.0
            phi_fixed = True

        else:
            phi_span = raw_span % two_pi
            phi_fixed = False

    indices = np.arange(n, dtype=float)

    if theta_fixed and phi_fixed:
        # Both angles fixed: every point is the same point.
        theta_values = np.full(n, theta_min, dtype=float)
        phi_values = np.full(n, phi_start, dtype=float)

    elif phi_fixed:
        # theta ranges, phi is fixed: n points equally spaced along a
        # meridian, including both poles of the range.
        theta_values = np.linspace(theta_min, theta_max, n)
        phi_values = np.full(n, phi_start, dtype=float)

    elif theta_fixed:
        # phi ranges, theta is fixed: n points equally spaced along a
        # latitude circle, starting at phi_start. phi is periodic, so
        # the far end of the interval is not given its own point when
        # the span is a full circle (it would duplicate phi_start).
        theta_values = np.full(n, theta_min, dtype=float)
        phi_values = (
            phi_start
            + (indices / n) * phi_span
        ) % two_pi

    else:
        # Both angles range: approximately equal-area coverage of the
        # spherical patch they bound. theta is equally spaced by area
        # (uniform in cos(theta)); phi advances by the golden angle so
        # points spiral across the patch instead of lining up in rows.
        u = (indices + 0.5) / n

        cos_theta = (
            np.cos(theta_min)
            + u * (np.cos(theta_max) - np.cos(theta_min))
        )
        theta_values = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        golden_ratio_conjugate = (np.sqrt(5.0) - 1.0) / 2.0
        v = np.mod((indices + 0.5) * golden_ratio_conjugate, 1.0)
        phi_values = (phi_start + v * phi_span) % two_pi

    sin_theta = np.sin(theta_values)

    unit_points = np.column_stack(
        (
            sin_theta * np.cos(phi_values),
            sin_theta * np.sin(phi_values),
            np.cos(theta_values),
        )
    )

    return center + radius * unit_points



class AxialFrame:
    """Convert vectors between axial-local and global coordinates."""

    @staticmethod
    def rotation_matrix(axis):
        """Return a rotation matrix whose local z-axis is aligned with axis."""
        axis = np.asarray(axis, dtype=float)

        if axis.shape != (3,):
            raise ValueError("axis must have shape (3,).")

        norm = np.linalg.norm(axis)

        if np.isclose(norm, 0.0):
            raise ValueError("axis cannot be the zero vector.")

        axis = axis / norm

        theta = np.arccos(
            np.clip(axis[2], -1.0, 1.0)
        )
        phi = np.arctan2(
            axis[1],
            axis[0],
        )

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return np.array([
            [
                cos_phi * cos_theta,
                -sin_phi,
                cos_phi * sin_theta,
            ],
            [
                sin_phi * cos_theta,
                cos_phi,
                sin_phi * sin_theta,
            ],
            [
                -sin_theta,
                0.0,
                cos_theta,
            ],
        ])

    @staticmethod
    def to_global(vector, axis, center=None):
        """Rotate an axial-local vector into global coordinates."""
        vector = np.asarray(vector, dtype=float)
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )

        return (
            center
            + AxialFrame.rotation_matrix(axis) @ vector
        )

    @staticmethod
    def to_local(vector, axis, center=None):
        """Rotate a global vector into axial-local coordinates."""
        vector = np.asarray(vector, dtype=float)
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )

        rotation = AxialFrame.rotation_matrix(axis)

        return rotation.T @ (vector - center)



def random_cone_direction(
    rng,
    direction,
    max_angle,
):
    """
    Generate a random unit direction within an angular distance of
    ``max_angle`` from a reference direction.

    The generated directions are uniformly distributed over the surface
    of the spherical cap.

    Parameters
    ----------
    rng : numpy.random.Generator-like
        Random number generator providing ``uniform()``.
    direction : array_like, shape (3,)
        Reference direction defining the cone axis.
    max_angle : float
        Maximum angle from ``direction``, in radians. Must lie within
        ``[0, pi]``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Random unit direction whose angle from ``direction`` is no greater
        than ``max_angle``.
    """
    direction = np.asarray(
        direction,
        dtype=float,
    )

    if direction.shape != (3,):
        raise ValueError(
            "direction must have shape (3,)."
        )

    norm = np.linalg.norm(direction)

    if np.isclose(norm, 0.0):
        raise ValueError(
            "direction cannot be the zero vector."
        )

    if not np.isscalar(max_angle):
        raise TypeError(
            "max_angle must be a scalar."
        )

    max_angle = float(max_angle)

    if not np.isfinite(max_angle):
        raise ValueError(
            "max_angle must be finite."
        )

    if not 0.0 <= max_angle <= np.pi / 2:
        raise ValueError(
            "max_angle must lie within [0, pi / 2]."
        )

    # Uniform solid-angle sampling inside the cone.
    theta = rng.random() * max_angle
    phi = rng.random() * 2.0 * np.pi

    local_direction = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )

    return AxialFrame.to_global(
        local_direction,
        direction,
    )




class EllipsoidalCoordinates:
    @staticmethod
    def normalized_radius(point, radii, center=None):
        """Return the normalized radius of a point relative to an ellipsoid."""
        point = np.asarray(point, dtype=float)
        radii = np.asarray(radii, dtype=float)
        center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)

        if point.shape != (3,):
            raise ValueError("point must have shape (3,).")
        if radii.shape != (3,):
            raise ValueError("radii must have shape (3,).")
        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(radii)) or not np.all(np.isfinite(center)):
            raise ValueError("point, radii, and center must contain finite values.")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be positive.")

        return np.linalg.norm((point - center) / radii)
    
    def depth(point, radii, center=None):
        """Return signed radial depth from the ellipsoid surface."""
        point = np.asarray(point, dtype=float)
        center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)

        normalized_radius = EllipsoidalCoordinates.normalized_radius(point, radii, center)
        distance_from_center = np.linalg.norm(point - center)

        if np.isclose(normalized_radius, 0.0):
            return float(np.min(radii))

        return -distance_from_center * (1.0 / normalized_radius - 1.0)

    @staticmethod
    def normal_direction(point, radii, center=None):
        """Return the outward unit normal to an ellipsoid at a given point."""
        point = np.asarray(point, dtype=float)
        radii = np.asarray(radii, dtype=float)
        center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)

        if point.shape != (3,):
            raise ValueError("point must have shape (3,).")
        if radii.shape != (3,):
            raise ValueError("radii must have shape (3,).")
        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(radii)) or not np.all(np.isfinite(center)):
            raise ValueError("point, radii, and center must contain finite values.")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be positive.")

        normal_direction = (point - center) / radii**2

        if np.isclose(np.linalg.norm(normal_direction), 0.0):
            raise ValueError("The ellipsoid normal is undefined at the center.")

        return to_unit_vector(normal_direction)

    @staticmethod
    def _prepare_radii(radii, dimension):
        """
        Normalize radii into an array of shape (dimension,).
        """
        if radii is None:
            radii = np.ones(dimension, dtype=float)

        elif np.isscalar(radii):
            radii = np.full(
                dimension,
                radii,
                dtype=float,
            )

        else:
            radii = np.asarray(radii, dtype=float)

        if radii.shape != (dimension,):
            raise ValueError(
                f"radii must be None, a scalar, or contain "
                f"exactly {dimension} values."
            )

        if np.any(radii <= 0):
            raise ValueError("All radii must be positive.")

        return radii

    @staticmethod
    def _prepare_center(center, dimension):
        """
        Normalize center into an array of shape (dimension,).
        """
        if center is None:
            return np.zeros(dimension, dtype=float)

        center = np.asarray(center, dtype=float)

        if center.shape != (dimension,):
            raise ValueError(
                f"center must contain exactly {dimension} values."
            )

        return center

    @staticmethod
    def from_cartesian(
        point,
        radii=None,
        center=None,
        dimension=None,
    ):
        """
        Convert Cartesian coordinates to normalized ellipsoidal coordinates.

        Parameters
        ----------
        point : array-like
            Cartesian coordinates [x, y] or [x, y, z].

        radii : None, scalar, or array-like, optional
            Ellipse or ellipsoid semi-axis lengths.

            None:
                All radii are set to 1.

            Scalar:
                The same radius is used for every axis.

            Array-like:
                One radius must be provided for each axis.

        center : array-like, optional
            Center of the ellipse or ellipsoid.
            Defaults to the origin.

        dimension : {2, 3}, optional
            Number of spatial dimensions. If omitted, inferred from `point`.

        Returns
        -------
        ndarray
            For 2D:
                [rho, theta]

            For 3D:
                [rho, theta, phi]

            theta:
                Azimuthal angle in [-pi, pi].

            phi:
                Polar angle measured from the positive z-axis in [0, pi].
        """
        point = np.asarray(point, dtype=float)

        if point.ndim != 1:
            raise ValueError("point must be a one-dimensional array.")

        if dimension is None:
            dimension = point.size

        if dimension not in (2, 3):
            raise ValueError("dimension must be either 2 or 3.")

        if point.shape != (dimension,):
            raise ValueError(
                f"point must contain exactly {dimension} values."
            )

        radii = EllipsoidalCoordinates._prepare_radii(
            radii,
            dimension,
        )

        center = EllipsoidalCoordinates._prepare_center(
            center,
            dimension,
        )

        normalized = (point - center) / radii

        rho = np.linalg.norm(normalized)

        phi = np.arctan2(
            normalized[1],
            normalized[0],
        )

        if dimension == 2:
            return np.array(
                [rho, phi],
                dtype=float,
            )

        radial_xy = np.hypot(
            normalized[0],
            normalized[1],
        )

        theta = np.arctan2(
            radial_xy,
            normalized[2],
        )

        return np.array(
            [rho, theta, phi],
            dtype=float,
        )

    @staticmethod
    def to_cartesian(
        coordinates,
        radii=None,
        center=None,
        dimension=None,
    ):
        """
        Convert normalized ellipsoidal coordinates to Cartesian coordinates.

        Parameters
        ----------
        coordinates : array-like
            For 2D:
                [rho, theta]

            For 3D:
                [rho, theta, phi]

        radii : None, scalar, or array-like, optional
            Ellipse or ellipsoid semi-axis lengths.

            None:
                All radii are set to 1.

            Scalar:
                The same radius is used for every axis.

            Array-like:
                One radius must be provided for each axis.

        center : array-like, optional
            Center of the ellipse or ellipsoid.
            Defaults to the origin.

        dimension : {2, 3}, optional
            Number of spatial dimensions. If omitted, inferred from
            `coordinates`.

        Returns
        -------
        ndarray
            Cartesian coordinates [x, y] or [x, y, z].
        """
        coordinates = np.asarray(coordinates, dtype=float)

        if coordinates.ndim != 1:
            raise ValueError(
                "coordinates must be a one-dimensional array."
            )

        if dimension is None:
            dimension = coordinates.size

        if dimension not in (2, 3):
            raise ValueError("dimension must be either 2 or 3.")

        if coordinates.shape != (dimension,):
            raise ValueError(
                f"coordinates must contain exactly {dimension} values."
            )

        radii = EllipsoidalCoordinates._prepare_radii(
            radii,
            dimension,
        )

        center = EllipsoidalCoordinates._prepare_center(
            center,
            dimension,
        )

        rho = coordinates[0]
        theta = coordinates[1]

        if dimension == 2:
            normalized = np.array([
                rho * np.cos(theta),
                rho * np.sin(theta),
            ])

        else:
            phi = coordinates[2]

            normalized = np.array([
                rho * np.sin(theta) * np.cos(phi),
                rho * np.sin(theta) * np.sin(phi),
                rho * np.cos(theta),
            ])

        return center + radii * normalized


