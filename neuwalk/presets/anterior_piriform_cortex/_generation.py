import json
from pathlib import Path
from ... import misc
from ...core.topology import SectionSynthesizer, connect_internal_branches
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
    #bifurcation_internal_density = all_params.pop("bifurcation_internal_density")

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
    spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)

    # create self-avoidance bias
    section_bias = biases.get_elongation("sibling_repulsion", 25.0, -2) +\
                biases.get_elongation("nonrelated_repulsion", 25.0, -2)

    # somatic repulsion
    somatic_bias = biases.get_elongation("root_repulsion", 750.0, -2)
    # compose the biases into the elongation bias
    elongation_bias = [
      (0.25, spatial_bias),
      (0.005, section_bias),
      (0.2, somatic_bias)
      ]
    
    # bifurcation biases
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)
    

    # initialize and synthesize the apical sections
    apic_synthesizer = _common.synthesize_section_tree(
        ret,
        seed,
        label='apical_dendrite',
        theta={0:0, "default":np.pi / 3},
        phi=0,
        axis_direction=np.array([0.0, 0.0, 1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
    )

    # spatial bias is a composition of truncated cones
    spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (-1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)    
    elongation_bias = [
      (0.25, spatial_bias),
      (0.005, section_bias),
      (0.2, somatic_bias)
      ]    
     
    # initialize and synthesize the basal sections
    basal_synthesizer = _common.synthesize_section_tree(
        ret,
        seed,
        label='basal_dendrite',
        theta=(0, np.pi / 2),
        phi=(0, 2 * np.pi),
        axis_direction=np.array([0.0, 0.0, -1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
        soma=apic_synthesizer.soma,
    )
    


    # return information
    ret['apical_dendrite']['morphology'] = apic_synthesizer
    ret['basal_dendrite']['morphology'] = basal_synthesizer
    ret['output'] = basal_synthesizer.soma
    
    return ret
