"""
Plot a neuron from an SWC file using neuwalk's own plot_morphology,
with a fixed color scheme (apical_dendrite=black, basal_dendrite=dark
red, apical_oblique=dark green), a fixed axis range and tick spacing
(so different neurons are directly, visually comparable across
separate plots rather than each auto-scaled to its own extent), and an
optional semi-transparent plane marking where the
anterior_piriform_cortex/neocortex pyramidal presets' spatial_bias
constrains dendritic growth.

Same range across plots: --axis-size now defaults to a fixed value
(500.0) instead of being derived per-neuron from the data, and is
centered on the origin (0, 0, 0) by default rather than each neuron's
own bounding-box center -- process_morphology's translate_sections
already puts the soma there, so every plot ends up with the EXACT same
absolute limits (-500 to 500 on every axis, by default) unless you
override --axis-size or pass --center-on-data to go back to centering
on that specific neuron's own data instead.

Right proportions: the same size is always applied to all three axes,
so the neuron is never stretched, whichever centering/size is in use.

Fixed tick spacing: ticks always land --tick-spacing units apart
(100.0 by default), regardless of the axis range in use, rather than
whatever spacing matplotlib's automatic locator would otherwise choose
for a given range.

--align-principal-axes rotates the neuron (via PCA on all its points)
so its longest spatial extent aligns with z, second longest with x,
and shortest with y -- a genuine geometric transformation of the
loaded points, not a change of viewing angle, so it composes normally
with --axis-size/--center-on-data/--show-bias-plane afterward.

Usage
-----
    python plot_neuron.py cell.swc
    python plot_neuron.py cell.swc --axis-size 300
    python plot_neuron.py cell.swc --center-on-data
    python plot_neuron.py cell.swc --tick-spacing 50
    python plot_neuron.py cell.swc --align-principal-axes
    python plot_neuron.py cell.swc --show-bias-plane
    python plot_neuron.py cell.swc --show-bias-plane --bias-plane-axis y --bias-plane-position 0
    python plot_neuron.py cell.swc --output cell.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from neuwalk.io import read_swc
from neuwalk.visualization.morphology import plot_morphology


SECTION_COLORS = {
    "apical_dendrite": "black",
    "basal_dendrite": "darkred",
    "apical_oblique": "darkgreen",
}


def set_axis_size(ax, size, center):
    """
    Force all three 3D axes to span [center - size, center + size] --
    the SAME size on every axis, so proportions stay correct -- rather
    than the size plot_morphology's own _set_equal_axes would derive
    from the plotted data.
    """
    center = np.asarray(center, dtype=float)

    ax.set_xlim(center[0] - size, center[0] + size)
    ax.set_ylim(center[1] - size, center[1] + size)
    ax.set_zlim(center[2] - size, center[2] + size)


def set_fixed_ticks(ax, spacing):
    """
    Force ticks on all three 3D axes to land exactly `spacing` units
    apart (e.g. ...,-100, 0, 100, 200,... for spacing=100), regardless
    of the current axis limits -- so tick spacing stays the same
    across different neurons and different --axis-size values, rather
    than whatever spacing matplotlib's automatic locator happens to
    pick for a given limit range.
    """
    ax.xaxis.set_major_locator(MultipleLocator(spacing))
    ax.yaxis.set_major_locator(MultipleLocator(spacing))
    ax.zaxis.set_major_locator(MultipleLocator(spacing))


def align_principal_axes(roots):
    """
    Rotate every point in every section so the neuron's own principal
    axes of spatial extent (via PCA on all its points) align with the
    plot's coordinate axes: the longest axis becomes z, the second
    longest becomes x, and the shortest becomes y. Mutates every
    section's points in place, through its own points setter (so
    validation still runs) -- this is a real geometric transformation
    of the loaded data, not just a change of viewing angle.

    A principal axis' sign is otherwise arbitrary (PCA/SVD singular
    vectors are only defined up to sign, which can flip unpredictably
    between neurons or even between runs) -- disambiguated here by
    flipping any axis whose points skew negative on median, so the
    bulk of the neuron consistently ends up on the positive side of
    each axis instead.
    """
    all_sections = [section for root in roots for section in root.subtree if len(section.points) > 0]

    if not all_sections:
        return

    all_points = np.vstack([np.asarray(section.points, dtype=float) for section in all_sections])

    center = all_points.mean(axis=0)
    centered = all_points - center

    # SVD on the centered point cloud: right singular vectors are the
    # principal axes, already ordered by decreasing variance (i.e.
    # decreasing extent along that direction).
    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)

    projections = centered @ principal_axes.T
    signs = np.sign(np.median(projections, axis=0))
    signs[signs == 0] = 1.0
    principal_axes = principal_axes * signs[:, np.newaxis]

    # principal_axes[0] is the longest axis, [1] the second, [2] the
    # shortest. Row i of `rotation` becomes the new axis i (x, y, z),
    # so rotation @ point = [longest . x_dir, ...] -- put the longest
    # axis in the row that produces the new z coordinate (row 2), the
    # second longest in the row for x (row 0), and the shortest in the
    # row for y (row 1).
    rotation = np.array([principal_axes[1], principal_axes[2], principal_axes[0]])

    for section in all_sections:
        points = np.asarray(section.points, dtype=float)
        section.points = (points - center) @ rotation.T


def add_bias_plane(ax, axis, position, color="orange", alpha=0.25):
    """
    Draw a semi-transparent plane orthogonal to the given axis
    ('x', 'y', or 'z'), at the given position along it, spanning the
    axis' current x/y/z limits.

    This represents the plane the anterior_piriform_cortex/neocortex
    pyramidal presets' spatial_bias constrains growth around: in
    _generation.py, spatial_bias is a pair of plane_boundary biases
    centered at y=+thickness and y=-thickness (thickness=25.0),
    together forming a slab that pushes dendrites back toward y=0 if
    they wander past either boundary -- so the default here
    (axis='y', position=0.0) marks the slab's own center plane, not
    either individual boundary.
    """
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

    if axis == "x":
        y = np.linspace(*ylim, 2)
        z = np.linspace(*zlim, 2)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, position)
    elif axis == "y":
        x = np.linspace(*xlim, 2)
        z = np.linspace(*zlim, 2)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, position)
    elif axis == "z":
        x = np.linspace(*xlim, 2)
        y = np.linspace(*ylim, 2)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, position)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False)

    # restore the limits plot_surface can otherwise expand
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("swc_file", help="Path to an .swc file.")
    parser.add_argument("--axis-size", type=float, default=500.0, help="Half-width of each axis, the same on every plot by default (default: 500.0).")
    parser.add_argument("--align-principal-axes", action="store_true", help="Rotate the neuron so its longest spatial extent aligns with z, second longest with x, and shortest with y, via PCA on its own points.")
    parser.add_argument("--center-on-data", action="store_true", help="Center the axis range on this neuron's own bounding-box center instead of the origin -- breaks direct comparability with other plots, but maximizes use of the range for this one.")
    parser.add_argument("--tick-spacing", type=float, default=100.0, help="Distance between axis ticks, the same on all three axes (default: 100.0).")
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument("--show-bias-plane", action="store_true", help="Overlay a semi-transparent orange plane marking the aPC/neocortex pyramidal presets' spatial_bias plane.")
    parser.add_argument("--bias-plane-axis", choices=("x", "y", "z"), default="y", help="Axis the bias plane is orthogonal to (default: y, matching _generation.py's plane_boundary bias).")
    parser.add_argument("--bias-plane-position", type=float, default=0.0, help="Position of the bias plane along --bias-plane-axis (default: 0.0, the slab's own center plane).")
    parser.add_argument("--output", default=None, help="Save the figure to this path instead of showing it interactively.")
    args = parser.parse_args()

    roots = read_swc(args.swc_file)

    if args.align_principal_axes:
        align_principal_axes(roots)

    ax = plot_morphology(roots, section_colors=SECTION_COLORS, linewidth=args.linewidth, show=False)

    if args.center_on_data:
        all_points = np.vstack([
            np.asarray(section.points, dtype=float)
            for root in roots
            for section in root.subtree
            if len(section.points) > 0
        ])
        center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    else:
        center = np.zeros(3)

    set_axis_size(ax, args.axis_size, center)

    if args.show_bias_plane:
        add_bias_plane(ax, args.bias_plane_axis, args.bias_plane_position)

    set_fixed_ticks(ax, args.tick_spacing)

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
