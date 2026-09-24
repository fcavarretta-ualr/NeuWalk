import json
from pathlib import Path
from ... import misc
from ...random import Random
from ...core.topology import SectionSynthesizer, connect_internal_branches, merge_trees
from ...synthesis import MorphologySynthesizer
from ...synthesis.morphology import biases
from .. import _common
import numpy as np

    
def generate(seed, cell_type, **kwargs):
    """Generate and return one neocortical pyramidal neuron."""
    
    # load the parameters
    path = Path(__file__).resolve().parent / (cell_type + ".parameters.corrected.json")
    all_params = json.loads(path.read_text())

    bin_size = kwargs.get("bin_size", 50.0)
    step_size = kwargs.get("step_size", 2.0)
    
    max_steps = kwargs.get("max_steps", 100)
    
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 3)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 100)
    max_total_attempts = kwargs.get("max_total_attempts", 5000)

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
    
    # spatial bias is a composition of truncated cones
    half_somatic_distance = {'semilunar':200, 'pyramidal':400}[cell_type]
    apical_spatial_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 0.]), (0., 0.), half_somatic_distance, -2)

    thickness = {'semilunar':25, 'pyramidal':50.0 } [cell_type]
    
    spatial_bias = biases.get_elongation("plane_boundary", np.array([0., thickness, 0.]), (np.pi / 2, np.pi / 2 * 3), thickness / 2, -2.0, resistance=True)  + \
                   biases.get_elongation("plane_boundary", np.array([0., -thickness, 0.]), (np.pi / 2, np.pi / 2), thickness / 2, -2.0, resistance=True)
    
    basal_spatial_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 0.]), (np.pi, 0.), 200, -2)

    # create self-avoidance bias

    
    #section_bias = biases.get_elongation("sibling_repulsion", 20, -2) + biases.get_elongation("nonrelated_repulsion", 10, -2)

    sibling_half_dist = 20 #{ "pyramidal":25, "semilunar":20 }[cell_type]
    non_sibiling_half_dist = 10 #{ "pyramidal":15, "semilunar":10 }[cell_type]
    
    section_bias = biases.get_elongation("sibling_repulsion", sibling_half_dist, -2) + biases.get_elongation("nonrelated_repulsion", non_sibiling_half_dist, -2)
    
    # somatic repulsion

    w_apic_section_bias = { "pyramidal": 0.0005, "semilunar":0.002 }[cell_type]
    w_apic_spatial_bias = { "pyramidal": 0.02, "semilunar":0.04 }[cell_type]

    # compose the biases into the elongation bias, one per label
    apical_elongation_bias = [
      (0.07, apical_spatial_bias),
      (w_apic_section_bias, section_bias),
      (w_apic_spatial_bias, spatial_bias)
      ]

    basal_elongation_bias = [
      (0.07, basal_spatial_bias),
      (0.00075, section_bias),
      (0.025, spatial_bias)
      ]
    
    # bifurcation biases
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)

    # merge apical's and basal's topologies so a single
    # MorphologySynthesizer can grow both labels together

    if cell_type == "pyramidal":
        merged_topology = merge_trees(
            ret['apical_dendrite']['topology'].soma,
            ret['basal_dendrite']['topology'].soma,
        )
    else:
        merged_topology = ret['apical_dendrite']['topology'].soma
        
    # generate apical first, and then basal dendrites
    merged_topology.set_order(0, labels='apical_dendrite')
    merged_topology.set_order(1, labels='basal_dendrite')


    elongation_random_weight = {
        "pyramidal":{"apical_dendrite":0.5, "basal_dendrite":2.5},
        "semilunar":{"apical_dendrite":2.5, "basal_dendrite":2.5}
        }[cell_type]

    
    synthesizer = MorphologySynthesizer(
        topology=merged_topology,
        rng=Random(seed),
        theta={1:0, "default":np.pi / 3},
        phi={1:0, "default":(0, 2 * np.pi)},
        axis_direction={
            'apical_dendrite': np.array([0.0, 0.0, 1.0]),
            'basal_dendrite': np.array([0.0, 0.0, -1.0]),
        },
        bifurcation_bias=bifurcation_bias,
        elongation_bias={'apical_dendrite': apical_elongation_bias, 'basal_dendrite': basal_elongation_bias},
        elongation_random_weight=elongation_random_weight,
        elongation_bias_weight=7.5,
        correction_type='somatic'
    )

    # synthesize() now runs both orders (apical, then basal) to
    # completion in a single call
    synthesizer.synthesize()



    # return information
    ret['synthesizer'] = synthesizer
    ret['output'] = synthesizer.soma
    
    return ret
