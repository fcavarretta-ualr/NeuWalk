import json
from pathlib import Path
from ... import misc
from ...profiles import NeuriteProfile, connect_internal_branches
from ...synthesis import TopologySynthesizer, MorphologySynthesizer
from ... import biases
import numpy as np


    
def generate(seed, **kwargs):
    """Generate and return one neocortical pyramidal neuron."""

    # load the parameters
    path = Path(__file__).resolve().parent / "parameters.json"
    all_params = json.loads(path.read_text())

    bin_size = kwargs.get("bin_size", 10.0)
    step_size = kwargs.get("step_size", 1.0)
    
    max_steps = kwargs.get("max_steps", 100)
    
    seed = kwargs.get("seed", 1234)
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 1.0)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 10)
    max_total_attempts = kwargs.get("max_total_attempts", 1000)


    # density of oblique branch points
    bifurcation_internal_density = all_params.pop("bifurcation_internal_density")

    # generate the profiles for each section type
    ret = {}
    
    for section_type, params in all_params.items():
      if verbose: print(f"Elaboration of {section_type}")
      
      if verbose: print(f"\tGenerating Branching-and-annihilating profile...", end="")
      topol_synthesizer = TopologySynthesizer(
            misc.Random(seed),
            step_size=step_size,
            section_type=section_type,
            with_soma=section_type != "apical_oblique",
            **params
        )

      topol_synthesizer.synthesize_progressive(
          n_std=n_std,
          max_attempts_per_window=max_attempts_per_window,
          max_total_attempts=max_total_attempts,
          verbose=verbose
      )

      ret[section_type] = {
        'topology':topol_synthesizer,
        }

      if verbose: print("done\n")

    # generate apical dendrites
    # connect obliques
    connect_internal_branches(ret['apical_oblique']['topology'].roots, ret['apical_dendrite']['topology'].soma.children, misc.Random(seed), bifurcation_internal_density, bin_size)
    
    # spatial bias is a composition of truncated cones
    spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (220., 0., 0.), (2.5, 2.5), (2.5, 2.5), 1, 1, strict=True) + \
                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 220.]), (50., 0., 0.), (2.5, 2.5), (25.0, 75.0), 1, 1, strict=True) + \
                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 270.]), (470., 0., 0.), (25.0, 75.0), (25.0, 75.0), 1, 1, strict=True) + \
                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 740.]), (160., 0., 0.), (25.0, 75.0), (150.0, 150.0), 1, 1, strict=True)

    # create self-avoidance bias
    dendritic_bias = biases.get_elongation("sibling_repulsion", 25.0, -2) +\
                biases.get_elongation("nonrelated_repulsion", 25.0, -2)

    # somatic repulsion
    somatic_bias = biases.get_elongation("root_repulsion", 750.0, -2)

    # plane boundary, push the distal apical dendrites to bend
    plane_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 900.]), (np.pi, 0.), 10, -2)

    # compose the biases into the elongation bias
    elongation_bias = [
      (0.25, spatial_bias),
      (0.005, dendritic_bias),
      (0.2, somatic_bias),
      (0.1, plane_bias)
      ]

    # bifurcation biases
    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)
    
    bifurcation_internal_bias = biases.get_bifurcation("internal_branch", np.pi / 2)  

    # initialize the synthesizer for apical dendrites
    apic_synthesizer = MorphologySynthesizer(
        root=ret['apical_dendrite']['topology'].soma,
        rng=misc.Random(seed),
        theta=0,
        phi=0,
        axis_direction=np.array([0.0, 0.0, 1.0]),
        bifurcation_bias=bifurcation_bias,
        bifurcation_internal_bias=bifurcation_internal_bias,
        elongation_bias=elongation_bias
    )

    # synthesize apical dendrites
    apic_synthesizer.synthesize()
    
    # synthesize basal dendrites   
    elongation_bias = [
      (0.005, dendritic_bias),
      (0.2, somatic_bias)
      ]
    
     
    basal_synthesizer = MorphologySynthesizer(
        root=ret['basal_dendrite']['topology'].soma,
        rng=misc.Random(seed),
        theta=(np.pi / 6, np.pi * 5 / 6),
        phi=(0, 2 * np.pi),
        axis_direction=np.array([0.0, 0.0, -1.0]),
        bifurcation_bias=bifurcation_bias,
        elongation_bias=elongation_bias,
    )

    basal_synthesizer.synthesize(soma=apic_synthesizer.soma)
    
    # re-use the same bias used for basal dendrites to generate oblique apical dendrites
    # the option is_root_like make the first branch as a root
    apic_synthesizer.synthesize(
      is_root_like=True,
      elongation_bias=elongation_bias,
      )


    # return information
    ret['apical_dendrite']['morphology'] = apic_synthesizer
    ret['basal_dendrite']['morphology'] = basal_synthesizer
    ret['output'] = basal_synthesizer.soma
    
    return ret
