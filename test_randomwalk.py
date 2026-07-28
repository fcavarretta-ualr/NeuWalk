#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np

from neuwalk.core.randomwalk import RandomWalk


def elongate(walk, steps):
    """Commit a fixed number of elongate steps."""
    for _ in range(steps):
        walk.elongate()
        walk.update_state()


def build_test_tree(rng, step_size, centrifugal=False):
    """Create a small branching random-walk tree."""
    root = RandomWalk(
        rng=rng,
        first_point=np.zeros(3),
        step_size=step_size,
        initial_direction=np.array([1.0, 0.0, 0.0]),
        centrifugal=centrifugal,
    )

    elongate(root, 8)

    root.bifurcate()
    left, right = root.update_state()

    elongate(left, 7)
    elongate(right, 5)

    left.bifurcate_internal()
    continuation, internal = left.update_state()

    elongate(continuation, 6)
    continuation.annihilate()
    continuation.update_state()

    internal = left.activate_internal_branch()
    elongate(internal, 4)
    internal.annihilate()
    internal.update_state()

    right.bifurcate()
    right_1, right_2 = right.update_state()

    elongate(right_1, 55)
    elongate(right_2, 6)

    right_1.annihilate()
    right_1.update_state()

    right_2.annihilate()
    right_2.update_state()

    return root


def plot_trajectories(root):
    """Plot the complete random-walk tree."""
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")

    all_points = []

    for walk in root._iter_walks():
        points = np.asarray(walk.points, dtype=float)

        if len(points) < 2:
            continue

        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
        )
        all_points.append(points)

    if all_points:
        points = np.vstack(all_points)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        radius = 0.5 * np.max(maximum - minimum)

        if np.isclose(radius, 0.0):
            radius = 0.5

        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Random-walk trajectories")

    plt.show()


def run_checks(root):
    """Run basic topology and state checks."""
    walks = list(root._iter_walks())

    assert not root.active
    assert len(root.children) == 2
    assert all(
        child.parent is root
        for child in root.children
    )
    assert all(
        len(walk.points) >= 2
        for walk in walks
    )
    assert all(
        walk.pending_move is None
        for walk in walks
    )

    print(f"walks: {len(walks)}")
    print(
        "points:",
        sum(len(walk.points) for walk in walks),
    )
    print(
        "active:",
        sum(walk.active for walk in walks),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test and plot the RandomWalk object."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--centrifugal",
        action="store_true",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    root = build_test_tree(
        rng=rng,
        step_size=args.step_size,
        centrifugal=args.centrifugal,
    )

    run_checks(root)
    plot_trajectories(root)


if __name__ == "__main__":
    main()
