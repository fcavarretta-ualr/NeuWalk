import numpy as np
import matplotlib.pyplot as plt
from ..core.section import Section, Neuron

def plot_morphology(
    roots,
    section_colors=None,
    ax=None,
    linewidth=1.5,
    show=True,
):
    """
    Plot a section and all its descendants in 3D.

    Parameters
    ----------
    roots : Section or sequence of Section
        Root(s) of the subtree(s) to plot.
    section_colors : dict, optional
        Mapping from labels to Matplotlib colors. Sections whose label
        is not present in the mapping are plotted in black.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axis.
    linewidth : float, default 1.5
        Line width.
    show : bool, default True
        Call ``plt.show()`` when True.

    Returns
    -------
    matplotlib.axes.Axes
        The 3D axis.
    """
    if isinstance(roots, Section):
        roots = [roots]

    if section_colors is None:
        section_colors = {}

    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

    all_points = []

    for root in roots:
        for section in root.subtree:

            points = np.asarray(section.points, dtype=float)

            if len(points) == 0:
                continue

            color = section_colors.get(
                section.label,
                "black",
            )


            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=linewidth,
            )

            all_points.append(points)

    if all_points:
        _set_equal_axes(
            ax,
            np.vstack(all_points),
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if show:
        plt.show()

    return ax


def _set_equal_axes(ax, points):
    """Set equal scaling for a 3D Matplotlib axis."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    center = 0.5 * (minimum + maximum)
    radius = 0.5 * np.max(maximum - minimum)

    if np.isclose(radius, 0.0):
        radius = 0.5

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
