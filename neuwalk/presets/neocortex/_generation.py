import json
from pathlib import Path
from ...random import Random
from ...core.topology import SectionSynthesizer, connect_internal_branches, merge_trees
from ...synthesis import MorphologySynthesizer
from ...synthesis.morphology import biases
from .. import _common
import numpy as np


    
def generate(seed, cell_type, **kwargs):
    """Generate and return one neocortical pyramidal neuron."""

    # load the parameters
    path = Path(__file__).resolve().parent / (cell_type + ".parameters.json")
    all_params = json.loads(path.read_text())

    bin_size = kwargs.get("bin_size", 10.0)
    step_size = kwargs.get("step_size", 1.0)
    
    max_steps = kwargs.get("max_steps", 100)
    
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 1.0)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 10)
    max_total_attempts = kwargs.get("max_total_attempts", 1000)


    # density of oblique branch points
    bifurcation_internal_density = all_params.pop("bifurcation_internal_density")

    # generate the profiles for each label
    ret = _common.synthesize_topologies(
        all_params,
        seed,
        step_size,
        n_std,
        max_attempts_per_window,
        max_total_attempts,
        verbose,
        with_soma=lambda label: label != "apical_oblique",
    )

    # generate apical sections
    # connect obliques (before merging, while apical's topology soma
    # still only has apical's own primary sections as children)
    connect_internal_branches(ret['apical_oblique']['topology'].roots, ret['apical_dendrite']['topology'].soma.children, Random(seed), bifurcation_internal_density, bin_size)
    
    # spatial bias is a composition of truncated cones
    spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (220., 0., 0.), (2.5, 2.5), (2.5, 2.5), 1, 1, strict=True) + \
                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 220.]), (50., 0., 0.), (2.5, 2.5), (25.0, 75.0), 1, 1, strict=True) + \
                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 270.]), (470., 0., 0.), (25.0, 75.0), (25.0, 75.0), 1, 1, strict=True) + \
                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 740.]), (160., 0., 0.), (25.0, 75.0), (150.0, 150.0), 1, 1, strict=True)

    # create self-avoidance bias
    section_bias = biases.get_elongation("sibling_repulsion", 25.0, -2) +\
                biases.get_elongation("nonrelated_repulsion", 25.0, -2)

    # somatic repulsion
    somatic_bias = biases.get_elongation("root_repulsion", 750.0, -2, consider_root_like=True)

    # plane boundary, push the distal apical sections to bend
    plane_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 900.]), (np.pi, 0.), 10, -2)

    # compose the biases into the elongation bias, one per label
    apical_elongation_bias = [
      (0.25, spatial_bias),
      (0.005, section_bias),
      (0.2, somatic_bias),
      (0.1, plane_bias)
      ]

    basal_elongation_bias = [
      (0.005, section_bias),
      (0.2, somatic_bias)
      ]

    # bifurcation biases
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)
    
    bifurcation_internal_bias = biases.get_bifurcation("internal_branch", np.pi / 2)  

    # merge apical's and basal's topologies so a single
    # MorphologySynthesizer can grow both labels together
    merged_topology = merge_trees(
        ret['apical_dendrite']['topology'].soma,
        ret['basal_dendrite']['topology'].soma,
    )

    # set explicit orders so each label gets its own synthesize() pass:
    # apical trunk, then basal trunk, then the grafted obliques
    merged_topology.set_order(0, labels='apical_dendrite')
    merged_topology.set_order(1, labels='basal_dendrite')
    merged_topology.set_order(2, labels='apical_oblique')

    synthesizer = MorphologySynthesizer(
        topology=merged_topology,
        rng=Random(seed),
        theta={'apical_dendrite': 0, 'basal_dendrite': (np.pi / 6, np.pi * 5 / 6)},
        phi={'apical_dendrite': 0, 'basal_dendrite': (0, 2 * np.pi)},
        axis_direction={
            'apical_dendrite': np.array([0.0, 0.0, 1.0]),
            'basal_dendrite': np.array([0.0, 0.0, -1.0]),
            'default': None,
        },
        bifurcation_bias=bifurcation_bias,
        bifurcation_internal_bias={'apical_dendrite': bifurcation_internal_bias, 'default': None},
        elongation_bias={
            'apical_dendrite': apical_elongation_bias,
            'basal_dendrite': basal_elongation_bias,
            # re-use basal's bias for obliques, same as before; needed
            # here (not just at the third synthesize() call) because a
            # child's label is resolved as soon as the internal branch
            # point is reached, during the very first (order 0) pass
            'apical_oblique': basal_elongation_bias,
        },
        # obliques have no axis_direction, so they keep the default
        # (no direction correction); only the primary apical and basal
        # trunks get pulled back toward the soma's axis.
        correction_type={
            'apical_dendrite': 'somatodendritic',
            'basal_dendrite': 'somatodendritic',
            'default': None,
        },
    )

    # synthesize() now runs every order (apical, basal, then the
    # grafted obliques) to completion in a single call
    synthesizer.synthesize()


    # return information
    ret['synthesizer'] = synthesizer
    ret['output'] = synthesizer.soma
    
    return ret
