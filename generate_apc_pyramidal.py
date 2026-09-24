"""
Generate anterior piriform cortex pyramidal neurons, with a generate()
function that lets you selectively disable elongation bias,
bifurcation bias, self-avoidance/somatic correction, and randomness --
independently of one another -- to isolate how much each contributes
to the final morphology.

This is a standalone copy of
neuwalk.presets.anterior_piriform_cortex._generation.generate for cell_type
"pyramidal", not a modification of the library: same topology synthesis,
same spatial/self-avoidance biases, same bifurcation bias, same
MorphologySynthesizer call -- just with new toggles wired into the
parameters _generation.py hardcodes.

elongation_bias itself is a COMPOSITION of three distinct terms (a
weighted sum, one entry per label in _generation.py's own
apical_elongation_bias/basal_elongation_bias lists) -- each is
independently toggleable here, rather than only being able to disable
elongation_bias as a whole:

- disable_directional_bias: the plane_boundary bias at the origin with
  no thickness (apical_spatial_bias/basal_spatial_bias in
  _generation.py's own naming) -- this is what gives apical dendrites
  a preference for growing "up" (theta=(0,0)) and basal dendrites a
  preference for growing "down" (theta=(pi,0)).
- disable_self_avoidance: sibling_repulsion + nonrelated_repulsion
  (section_bias in _generation.py) -- keeps sections from overlapping
  or clustering into each other.
- disable_layer_bias: the pair of plane_boundary biases at y=+/-
  thickness (spatial_bias in _generation.py) -- the "slab" that keeps
  dendrites within a layer, the same one plot_neuron.py's
  --show-bias-plane visualizes.

Disabling all three removes elongation_bias entirely (same effect the
old, single disable_elongation_bias toggle had).

What each of the other toggles does
-------------------------------------
- disable_bifurcation_bias: passes bifurcation_bias=None instead of
  the radial_torsion bias, so a bifurcation's two children get no
  directed pull apart from each other beyond the base angular sampling.
- disable_correction: passes correction_type=None instead of
  'somatic', removing the corrective pull back toward a consistent
  outward direction from the soma -- sections may then wander further
  from strictly radial growth.
- disable_randomness: sets elongation_random_weight=0 instead of 0.5,
  removing the random component from every elongation step -- growth
  direction is then driven only by bias and correction (unless those
  are also disabled).

Disabling every bias, correction, and randomness toggle at once
collapses elongation to a fixed direction that never changes after the
first step (nothing left to alter it), which is degenerate but is a
legitimate way to inspect the topology's own shape in isolation from
any morphology-level geometry decisions.

Performance note
-----------------
Every toggle EXCEPT disable_self_avoidance is substantially SLOWER
than the default when used alone (tens of seconds instead of a
fraction of one, in testing) -- disable_directional_bias,
disable_layer_bias, disable_bifurcation_bias, disable_correction, and
disable_randomness were all measured in the 37-44 second range
individually. Each removes a different influence that normally keeps
sections spatially differentiated from one another (a directional
preference, a layer constraint, a push apart at bifurcations, a pull
back toward radial growth, or random perturbation, respectively) while
leaving self-avoidance (sibling_repulsion/nonrelated_repulsion) fully
active. With sections clustering closer together than usual,
self-avoidance's own pairwise distance computation has substantially
more nearby sections to resolve on every step, which is where the
slowdown comes from directly (confirmed by interrupting a slow run and
inspecting the stack: it's inside
neuwalk.synthesis.morphology.biases.elongation.repulsion.section_repulsion,
not stuck in any retry or validation loop).

Passing disable_self_avoidance=True alongside ANY of the toggles above
removes this cost directly, confirmed for several combinations (3-4
seconds each, versus 37-44 alone): there's nothing expensive left for
sections to need to avoid once self-avoidance itself is gone.

Usage
-----
As a library:

    from generate_apc_pyramidal import generate
    result = generate(seed=0)
    result = generate(seed=0, disable_randomness=True)
    result = generate(seed=0, disable_directional_bias=True, disable_bifurcation_bias=True)
    result = generate(seed=0, disable_randomness=True, disable_self_avoidance=True)  # avoids the slowdown above

From the command line:

    python generate_apc_pyramidal.py --seed 0 --output cell.swc
    python generate_apc_pyramidal.py --seed 0 --output cell.swc --disable-randomness
"""

import argparse
import json
from pathlib import Path

import numpy as np

from neuwalk import misc
from neuwalk.random import Random
from neuwalk.core.topology import merge_trees
from neuwalk.io import write_swc
from neuwalk.synthesis import MorphologySynthesizer
from neuwalk.synthesis.morphology import biases
from neuwalk.presets import _common
import neuwalk.presets.anterior_piriform_cortex as _apc_package


def generate(
    seed,
    disable_directional_bias=False,
    disable_self_avoidance=False,
    disable_layer_bias=False,
    disable_bifurcation_bias=False,
    disable_correction=False,
    disable_randomness=False,
    **kwargs,
):
    """
    Generate one anterior piriform cortex pyramidal neuron.

    Parameters
    ----------
    seed : int
        Random seed.
    disable_directional_bias : bool, default False
        If True, removes the plane_boundary bias at the origin that
        gives apical dendrites a preference for growing "up" and
        basal dendrites a preference for growing "down".
    disable_self_avoidance : bool, default False
        If True, removes sibling_repulsion + nonrelated_repulsion, so
        sections no longer avoid overlapping or clustering into each
        other. See the module docstring's Performance note: this also
        removes the slowdown every OTHER toggle in this function
        otherwise causes when used alone.
    disable_layer_bias : bool, default False
        If True, removes the pair of plane_boundary biases at
        y=+/-thickness that keep dendrites within a layer (the same
        plane plot_neuron.py's --show-bias-plane visualizes).
    disable_bifurcation_bias : bool, default False
        If True, bifurcating children get no directed bias.
    disable_correction : bool, default False
        If True, no corrective pull toward a consistent outward
        direction is applied during elongation.
    disable_randomness : bool, default False
        If True, the random component of elongation direction is
        removed (elongation_random_weight=0).
    **kwargs
        Forwarded to synthesize_topologies: bin_size, step_size,
        n_std, max_attempts_per_window, max_total_attempts, verbose.

    Returns
    -------
    dict
        Same structure as _generation.generate: per-label topology
        entries, plus 'synthesizer' (the MorphologySynthesizer) and
        'output' (the synthesized soma).
    """
    # locate pyramidal.parameters.corrected.json via the actual
    # installed preset package, not this script's own location -- this
    # script is meant to be run from anywhere, unlike the library's own
    # _generation.py, which can safely assume it sits next to the file
    # it loads
    path = Path(_apc_package.__file__).resolve().parent / "pyramidal.parameters.corrected.json"
    all_params = json.loads(path.read_text())

    bin_size = kwargs.get("bin_size", 50.0)
    step_size = kwargs.get("step_size", 2.0)
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 3)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 50)
    max_total_attempts = kwargs.get("max_total_attempts", 1000)

    # generate the profiles for each label
    ret = _common.synthesize_topologies(
        all_params,
        seed,
        step_size,
        n_std,
        max_attempts_per_window,
        max_total_attempts,
        verbose,
    )

    # each of the three elongation_bias components below is built (or
    # left as None) independently, then composed into one weighted
    # list per label -- matching _generation.py's own structure and
    # weights exactly, just with each term individually skippable
    apical_spatial_bias = None if disable_directional_bias else \
        biases.get_elongation("plane_boundary", np.array([0., 0., 0.]), (0., 0.), None, None)

    thickness = 25.0

    if disable_layer_bias:
        spatial_bias = None
    else:
        spatial_bias = biases.get_elongation("plane_boundary", np.array([0., thickness, 0.]), (np.pi / 2, np.pi / 2 * 3), thickness / 2, -1.0, resistance=True) + \
                       biases.get_elongation("plane_boundary", np.array([0., -thickness, 0.]), (np.pi / 2, np.pi / 2), thickness / 2, -1.0, resistance=True)

    basal_spatial_bias = None if disable_directional_bias else \
        biases.get_elongation("plane_boundary", np.array([0., 0., 0.]), (np.pi, 0.), None, None)

    section_bias = None if disable_self_avoidance else \
        (biases.get_elongation("sibling_repulsion", 20, -2) + biases.get_elongation("nonrelated_repulsion", 10, -2))

    # compose the biases into the elongation bias, one per label --
    # a term whose bias came out None (that component was disabled)
    # is dropped entirely, and a label left with no terms at all gets
    # elongation_bias=None for that label, matching how
    # MorphologySynthesizer expects "no bias" to be expressed
    apical_terms = [(0.04, apical_spatial_bias), (0.001, section_bias), (0.015, spatial_bias)]
    basal_terms = [(0.04, basal_spatial_bias), (0.001, section_bias), (0.015, spatial_bias)]

    apical_elongation_bias = [(weight, bias) for weight, bias in apical_terms if bias is not None] or None
    basal_elongation_bias = [(weight, bias) for weight, bias in basal_terms if bias is not None] or None

    # bifurcation bias
    bifurcation_bias = None if disable_bifurcation_bias else biases.get_bifurcation("radial_torsion", np.pi / 3)

    correction_type = None if disable_correction else "somatic"

    elongation_random_weight = 0.0 if disable_randomness else 0.5

    # merge apical's and basal's topologies so a single
    # MorphologySynthesizer can grow both labels together
    merged_topology = merge_trees(
        ret["apical_dendrite"]["topology"].soma,
        ret["basal_dendrite"]["topology"].soma,
    )

    # generate apical first, and then basal dendrites
    merged_topology.set_order(0, labels="apical_dendrite")
    merged_topology.set_order(1, labels="basal_dendrite")

    synthesizer = MorphologySynthesizer(
        topology=merged_topology,
        rng=Random(seed),
        theta={1: 0, "default": np.pi / 6},
        phi={1: 0, "default": (0, 2 * np.pi)},
        axis_direction={
            "apical_dendrite": np.array([0.0, 0.0, 1.0]),
            "basal_dendrite": np.array([0.0, 0.0, -1.0]),
        },
        bifurcation_bias=bifurcation_bias,
        elongation_bias={"apical_dendrite": apical_elongation_bias, "basal_dendrite": basal_elongation_bias},
        elongation_random_weight=elongation_random_weight,
        elongation_bias_weight=2.0,
        correction_type=correction_type,
    )

    # synthesize() runs both orders (apical, then basal) to completion
    # in a single call
    synthesizer.synthesize()

    ret["synthesizer"] = synthesizer
    ret["output"] = synthesizer.soma

    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="Random seed for the neuron.")
    parser.add_argument("--output", required=True, help="Path to write the generated neuron as an .swc file.")
    parser.add_argument("--disable-directional-bias", action="store_true", help="Removes the 'grow up'/'grow down' preference for apical/basal dendrites.")
    parser.add_argument("--disable-self-avoidance", action="store_true", help="Removes sibling/nonrelated repulsion. See the module docstring's Performance note: combine with this to avoid the slowdown every other toggle here causes when used alone.")
    parser.add_argument("--disable-layer-bias", action="store_true", help="Removes the layer-constraining plane_boundary bias (see plot_neuron.py --show-bias-plane).")
    parser.add_argument("--disable-bifurcation-bias", action="store_true")
    parser.add_argument("--disable-correction", action="store_true")
    parser.add_argument("--disable-randomness", action="store_true")
    parser.add_argument("--bin-size", type=float, default=50.0)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--n-std", type=float, default=3.0)
    parser.add_argument("--max-attempts-per-window", type=int, default=50)
    parser.add_argument("--max-total-attempts", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = generate(
        args.seed,
        disable_directional_bias=args.disable_directional_bias,
        disable_self_avoidance=args.disable_self_avoidance,
        disable_layer_bias=args.disable_layer_bias,
        disable_bifurcation_bias=args.disable_bifurcation_bias,
        disable_correction=args.disable_correction,
        disable_randomness=args.disable_randomness,
        bin_size=args.bin_size,
        step_size=args.step_size,
        n_std=args.n_std,
        max_attempts_per_window=args.max_attempts_per_window,
        max_total_attempts=args.max_total_attempts,
        verbose=args.verbose,
    )

    write_swc(args.output, [result["output"]])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
