import numpy as np

import copy


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
    points = np.asarray(points, dtype=float)
    source = np.asarray(source, dtype=float)

    if target is None:
        target = np.zeros(3, dtype=float)

    target = np.asarray(target, dtype=float)

    if points.shape[-1] != 3:
        raise ValueError("points must have final dimension 3.")

    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("source and target must have shape (3,).")

    return points + target - source

class Random:
    """Small wrapper around a random-number generator."""

    def __init__(self, seed=None):
        """
        Parameters
        ----------
        seed : int, optional
            Seed used to initialize the NumPy random-number generator.
        """
        self._rng = np.random.default_rng(seed)

    def random(self, n=None):
        """
        Generate random values in the interval [0, 1).

        Parameters
        ----------
        n : int, optional
            Number of values to generate. When omitted, return a scalar.

        Returns
        -------
        float or numpy.ndarray
            One random value or an array with shape ``(n,)``.
        """
        if n is None:
            return float(self._rng.random())

        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer or None.")

        if n < 0:
            raise ValueError("n cannot be negative.")

        return self._rng.random(n)

    def clone(self):
        """
        Return a new Random object with the same internal state.

        The clone produces the same future sequence as the original until
        either object is advanced independently.
        """
        clone = self.__class__()
        clone._rng.bit_generator.state = copy.deepcopy(
            self._rng.bit_generator.state
        )
        return clone

def slerp(u, v, t):
    """
    Interpolate between two directions on the unit sphere.

    Parameters
    ----------
    u, v : array-like
        Input 3D directions.
    t : float
        Interpolation parameter between 0 and 1.

    Returns
    -------
    numpy.ndarray
        Interpolated unit direction.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be between 0 and 1.")

    u = _normalize(u)
    v = _normalize(v)

    dot = np.clip(np.dot(u, v), -1.0, 1.0)

    # Nearly parallel vectors.
    if np.isclose(dot, 1.0):
        return _normalize(
            (1.0 - t) * u + t * v
        )

    # Opposite vectors do not define a unique shortest spherical path.
    if np.isclose(dot, -1.0):
        orthogonal = _orthogonal_unit_vector(u)

        return _normalize(
            np.cos(np.pi * t) * u
            + np.sin(np.pi * t) * orthogonal
        )

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    result = (
        np.sin((1.0 - t) * theta) / sin_theta * u
        + np.sin(t * theta) / sin_theta * v
    )

    return _normalize(result)


def _normalize(vector):
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


def _orthogonal_unit_vector(vector):
    """Return a deterministic unit vector orthogonal to a 3D vector."""
    axis = np.zeros(3)
    axis[np.argmin(np.abs(vector))] = 1.0

    return _normalize(
        np.cross(vector, axis)
    )



def hill(x, half, exponent=1.0, minimum=0.0, maximum=1.0):
    """
    Generalized Hill function for scalar or array inputs.

    Negative exponents produce a decreasing Hill function.
    """
    x = np.asarray(x, dtype=float)

    if half <= 0:
        raise ValueError("half must be positive.")

    if np.any(x < 0):
        raise ValueError("x must be nonnegative.")

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


class EllipsoidalCoordinates:
    """Convert between Cartesian and ellipsoidal polar coordinates."""

    def __init__(
        self,
        semi_axes,
        center=None,
        rotation=None,
    ):
        """
        Initialize the ellipsoidal coordinate system.

        Parameters
        ----------
        semi_axes : array-like
            Ellipsoid semi-axes ``[a, b, c]``.
        center : array-like, optional
            Ellipsoid center. Defaults to the origin.
        rotation : array-like, optional
            Rotation matrix from local ellipsoid coordinates to global
            coordinates. Defaults to the identity matrix.
        """
        semi_axes = np.asarray(semi_axes, dtype=float)

        if semi_axes.shape != (3,):
            raise ValueError(
                "semi_axes must have shape (3,)."
            )

        if np.any(semi_axes <= 0):
            raise ValueError(
                "All semi-axes must be positive."
            )

        if center is None:
            center = np.zeros(3, dtype=float)

        center = np.asarray(center, dtype=float)

        if center.shape != (3,):
            raise ValueError(
                "center must have shape (3,)."
            )

        if rotation is None:
            rotation = np.eye(3, dtype=float)

        rotation = np.asarray(rotation, dtype=float)

        if rotation.shape != (3, 3):
            raise ValueError(
                "rotation must have shape (3, 3)."
            )

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
        ):
            raise ValueError(
                "rotation must be an orthogonal matrix."
            )

        self.semi_axes = semi_axes.copy()
        self.center = center.copy()
        self.rotation = rotation.copy()

    def to_polar(self, point):
        """
        Convert Cartesian points to ellipsoidal polar coordinates.

        Parameters
        ----------
        point : array-like
            Point or array of points with final dimension 3.

        Returns
        -------
        tuple
            ``(rho, theta, phi)``, where ``theta`` is the polar angle from
            the positive local z-axis and ``phi`` is the azimuthal angle.
        """
        point = np.asarray(point, dtype=float)

        if point.shape[-1] != 3:
            raise ValueError(
                "point must have final dimension 3."
            )

        local = self.to_local(point)
        scaled = local / self.semi_axes

        x = scaled[..., 0]
        y = scaled[..., 1]
        z = scaled[..., 2]

        rho = np.linalg.norm(
            scaled,
            axis=-1,
        )

        theta = np.zeros_like(rho, dtype=float)

        nonzero = ~np.isclose(rho, 0.0)

        theta[nonzero] = np.arccos(
            np.clip(
                z[nonzero] / rho[nonzero],
                -1.0,
                1.0,
            )
        )

        phi = np.arctan2(y, x)

        return rho, theta, phi

    def from_polar(
        self,
        rho,
        theta,
        phi,
    ):
        """
        Convert ellipsoidal polar coordinates to Cartesian coordinates.

        Parameters
        ----------
        rho : float or array-like
            Normalized ellipsoidal radius.
        theta : float or array-like
            Polar angle from the positive local z-axis.
        phi : float or array-like
            Azimuthal angle in the local xy-plane.

        Returns
        -------
        numpy.ndarray
            Cartesian points with final dimension 3.
        """
        rho = np.asarray(rho, dtype=float)
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)

        rho, theta, phi = np.broadcast_arrays(
            rho,
            theta,
            phi,
        )

        if np.any(rho < 0):
            raise ValueError(
                "rho cannot be negative."
            )

        scaled = np.stack(
            [
                rho * np.sin(theta) * np.cos(phi),
                rho * np.sin(theta) * np.sin(phi),
                rho * np.cos(theta),
            ],
            axis=-1,
        )

        local = scaled * self.semi_axes

        return self.to_global(local)

    def normalized_radius(self, point):
        """
        Return the normalized ellipsoidal radius.

        A value of 1 indicates a point on the ellipsoid.
        """
        rho, _, _ = self.to_polar(point)
        return rho

    def is_inside(
        self,
        point,
        tolerance=1e-12,
    ):
        """Return whether points are inside or on the ellipsoid."""
        return (
            self.normalized_radius(point)
            <= 1.0 + tolerance
        )

    def project_to_surface(self, point):
        """
        Project points radially onto the ellipsoid surface.

        The projection follows a ray from the ellipsoid center.
        """
        rho, theta, phi = self.to_polar(point)

        if np.any(np.isclose(rho, 0.0)):
            raise ValueError(
                "The ellipsoid center has no unique radial "
                "projection onto the surface."
            )

        return self.from_polar(
            rho=np.ones_like(rho),
            theta=theta,
            phi=phi,
        )

    def surface_normal(self, point):
        """
        Return the outward unit normal of the ellipsoid.

        Points are expected to lie on the ellipsoid surface.
        """
        local = self.to_local(point)

        normal_local = (
            local / self.semi_axes**2
        )

        normal_global = (
            normal_local @ self.rotation.T
        )

        norm = np.linalg.norm(
            normal_global,
            axis=-1,
            keepdims=True,
        )

        if np.any(np.isclose(norm, 0.0)):
            raise ValueError(
                "The surface normal is undefined at the center."
            )

        return normal_global / norm

    def centrifugal_direction(self, point):
        """
        Return the radial direction from the ellipsoid center.

        This follows the center-to-point ray and is generally different
        from the ellipsoid surface normal.
        """
        point = np.asarray(point, dtype=float)
        vector = point - self.center

        norm = np.linalg.norm(
            vector,
            axis=-1,
            keepdims=True,
        )

        if np.any(np.isclose(norm, 0.0)):
            raise ValueError(
                "The radial direction is undefined at the center."
            )

        return vector / norm

    def to_local(self, point):
        """Transform global points to local ellipsoid coordinates."""
        point = np.asarray(point, dtype=float)

        return (
            point - self.center
        ) @ self.rotation

    def to_global(self, point):
        """Transform local ellipsoid points to global coordinates."""
        point = np.asarray(point, dtype=float)

        return (
            point @ self.rotation.T
            + self.center
        )
