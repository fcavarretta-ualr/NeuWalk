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

<<<<<<< HEAD
    bin_size = kwargs.get("bin_size", 10.0)
    step_size = kwargs.get("step_size", 2.5)
=======
    bin_size = kwargs.get("bin_size", 50.0)
    step_size = kwargs.get("step_size", 2.0)
>>>>>>> 21a7a56 (last version)
    
    max_steps = kwargs.get("max_steps", 100)
    
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 3)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 25)
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
    
    # spatial bias is a composition of truncated cones
    apical_spatial_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 0.]), (0., 0.), None, None)

<<<<<<< HEAD
    thickness = {'semilunar':200, 'pyramidal':200}[cell_type]
    repulsion_weight = {'semilunar':0.002, 'pyramidal':0.002}[cell_type]
    
    spatial_bias = biases.get_elongation("plane_boundary", np.array([0., thickness, 0.]), (np.pi / 2, np.pi / 2 * 3), thickness / 2, -2.0)  + \
                   biases.get_elongation("plane_boundary", np.array([0., -thickness, 0.]), (np.pi / 2, np.pi / 2), thickness / 2, -2.0)
=======
    thickness = 25.0
    
    spatial_bias = biases.get_elongation("plane_boundary", np.array([0., thickness, 0.]), (np.pi / 2, np.pi / 2 * 3), thickness / 2, -1.0, resistance=True)  + \
                   biases.get_elongation("plane_boundary", np.array([0., -thickness, 0.]), (np.pi / 2, np.pi / 2), thickness / 2, -1.0, resistance=True)
>>>>>>> 21a7a56 (last version)
    
    basal_spatial_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 0.]), (np.pi, 0.), None, None)

    # create self-avoidance bias
<<<<<<< HEAD
    section_bias = biases.get_elongation("sibling_repulsion", 25, -2) + biases.get_elongation("nonrelated_repulsion", 15, -2)
=======
    section_bias = biases.get_elongation("sibling_repulsion", 20, -2) + biases.get_elongation("nonrelated_repulsion", 10, -2)
>>>>>>> 21a7a56 (last version)
    # somatic repulsion

    # compose the biases into the elongation bias, one per label
    apical_elongation_bias = [
<<<<<<< HEAD
      (0.75, apical_spatial_bias),
      (repulsion_weight, section_bias),
      (0.3, spatial_bias)
      ]

    basal_elongation_bias = [
      (0.75, basal_spatial_bias),
      (0.002, section_bias),
      (0.3, spatial_bias)
=======
      (0.04, apical_spatial_bias),
      (0.001, section_bias),
      (0.015, spatial_bias)
      ]

    basal_elongation_bias = [
      (0.04, basal_spatial_bias),
      (0.001, section_bias),
      (0.015, spatial_bias)
>>>>>>> 21a7a56 (last version)
      ]
    
    # bifurcation biases
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)

    # merge apical's and basal's topologies so a single
    # MorphologySynthesizer can grow both labels together
    merged_topology = merge_trees(
        ret['apical_dendrite']['topology'].soma,
        ret['basal_dendrite']['topology'].soma,
    )
<<<<<<< HEAD

=======
    
>>>>>>> 21a7a56 (last version)
    # generate apical first, and then basal dendrites
    merged_topology.set_order(0, labels='apical_dendrite')
    merged_topology.set_order(1, labels='basal_dendrite')
    
    synthesizer = MorphologySynthesizer(
        topology=merged_topology,
        rng=Random(seed),
<<<<<<< HEAD
        theta={1:0, "default":np.pi / 3},
=======
        theta={1:0, "default":np.pi / 6},
>>>>>>> 21a7a56 (last version)
        phi={1:0, "default":(0, 2 * np.pi)},
        axis_direction={
            'apical_dendrite': np.array([0.0, 0.0, 1.0]),
            'basal_dendrite': np.array([0.0, 0.0, -1.0]),
        },
        bifurcation_bias=bifurcation_bias,
        elongation_bias={'apical_dendrite': apical_elongation_bias, 'basal_dendrite': basal_elongation_bias},
<<<<<<< HEAD
        elongation_random_weight=1.5,
        elongation_bias_weight=5,
        correction_type='somatodendritic',
=======
        elongation_random_weight=0.5,
        elongation_bias_weight=2.0,
        correction_type='somatodendritic'
>>>>>>> 21a7a56 (last version)
    )

    # synthesize() now runs both orders (apical, then basal) to
    # completion in a single call
    synthesizer.synthesize()



    # return information
    ret['synthesizer'] = synthesizer
    ret['output'] = synthesizer.soma
    
    return ret
