import json
from pathlib import Path
from ... import misc
from ...profiles import NeuriteProfile, connect_internal_branches
from ...synthesis import TopologySynthesizer, MorphologySynthesizer
from ... import biases
import numpy as np

radii = np.array([2000, 1250, 1250], dtype=float)
epl_depth = 300
glom_radius = 50.

def merge_profiles(profile_roots):
    # for non oblique, roots are attached to the soma          
    profile_soma = NeuriteProfile(1, section_type="soma")
    for root in profile_roots:
        root.connect(profile_soma, relation="parent")
    return profile_soma
    
def generate_apical(seed, step_size, soma_position, glom_position, axis_direction):
    topol_synthesizer = TopologySynthesizer(
        misc.Random(seed),
        step_size=step_size,
        section_type="apical_dendrite",
        sholl_plot={'mean':[1,1], 'std':[0,0]},
        bin_size=np.linalg.norm(soma_position-glom_position),
        primary_count_range={"min":1, "max":1}
    )

    topol_synthesizer.synthesize()


    # initializa the synthesizer for apical dendrites
    apical_synthesizer = MorphologySynthesizer(
        root=merge_profiles(topol_synthesizer.roots),
        rng=misc.Random(seed=seed),
        theta=0,
        phi=0,
        origin=soma_position,
        axis_direction=axis_direction,
        elongation_bias=biases.get_elongation("attraction", None, None, glom_position)
    )

    # synthesize apical dendrites
    soma = apical_synthesizer.synthesize()

    # synthesize the tuft dendrites
    topol_synthesizer = TopologySynthesizer(
        misc.Random(seed),
        step_size=step_size,
        section_type="apical_dendrite",
        sholl_plot={'mean':[5,10,20,40,80,80,40,20,10,5,0], 'std':[0,5,10,20,40,40,20,10,5,2.5,0]},
        bin_size=10,
        primary_count_range={"min":4, "max":6}
    )

    topol_synthesizer.synthesize()


    center = apical_synthesizer.soma.children[0].points[-1] + axis_direction * glom_radius

    # initializa the synthesizer for apical dendrites
    tuft_synthesizer = MorphologySynthesizer(
        root=merge_profiles(topol_synthesizer.roots),
        rng=misc.Random(seed=seed),
        theta=0,
        phi=0,
        origin=apical_synthesizer.soma.children[0].points[-1],
        axis_direction=axis_direction,
        elongation_bias=biases.get_elongation("ellipsoid_boundary", np.zeros(3, dtype=float) + 50.0, 1, -1, orientation='in', center=center)
    )

    # synthesize apical dendrites
    tuft_origin = tuft_synthesizer.synthesize()
    for ch in tuft_origin.children:
        ch.disconnect_from_parent()
        ch.connect(soma, relation="parent")
    return soma

    
def generate(seed, cell_type, **kwargs):
    """Generate and return one neocortical pyramidal neuron."""
    global radii, epl_depth

    if cell_type == "mitral":
        soma_layer = radii - epl_depth
        primary_theta = np.pi / 6
        spatial_bias = biases.get_elongation("ellipsoid_boundary", radii, epl_depth / 2, -1, orientation='in') + \
                       biases.get_elongation("ellipsoid_boundary", soma_layer, 75., -1, orientation='out')
    elif cell_type == "middle_tufted":
        soma_layer = radii - epl_depth / 2
        primary_theta = np.pi / 2
        spatial_bias = biases.get_elongation("ellipsoid_boundary", radii, 1.0, -1, orientation='in') + \
                       biases.get_elongation("ellipsoid_boundary", soma_layer, 50., -1, orientation='out')
    else:
        raise ValueError("Unknown cell type.")

    

  

    
    # load the parameters
    path = Path(__file__).resolve().parent / (cell_type + ".parameters.json")
    all_params = json.loads(path.read_text())

    bin_size = kwargs.get("bin_size", 10.0)
    step_size = kwargs.get("step_size", 1.0)
    
    max_steps = kwargs.get("max_steps", 100)
    
    seed = kwargs.get("seed", 1234)
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 1.0)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 10)
    max_total_attempts = kwargs.get("max_total_attempts", 1000)


    theta = kwargs.get("theta", 0.)
    phi = kwargs.get("phi", 0.)
    
    # calculate soma and glomerulus location
    soma_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, theta, phi), soma_layer)
    glom_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, theta, phi), radii)
    axis_direction = misc.to_unit_vector(misc.EllipsoidalCoordinates.to_cartesian((1, theta, phi), radii))

    
    # density of oblique branch points
    #bifurcation_internal_density = all_params.pop("bifurcation_internal_density")

    # generate the profiles for each section type
    ret = {}
    
    for section_type, params in all_params.items():
      if verbose: print(f"Elaboration of {section_type}")
      
      if verbose: print(f"\tGenerating Branching-and-annihilating profile...", end="")
      topol_synthesizer = TopologySynthesizer(
            misc.Random(seed),
            step_size=step_size,
            section_type=section_type,
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

##    # generate apical dendrites
##    # connect obliques
##    connect_internal_branches(ret['apical_oblique']['topology'].roots, ret['apical_dendrite']['topology'].roots, misc.Random(seed), bifurcation_internal_density, bin_size)
##    
##    # spatial bias is a composition of truncated cones
##    spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (220., 0., 0.), (2.5, 2.5), (2.5, 2.5), 1, 1, strict=True) + \
##                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 220.]), (50., 0., 0.), (2.5, 2.5), (25.0, 75.0), 1, 1, strict=True) + \
##                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 270.]), (470., 0., 0.), (25.0, 75.0), (25.0, 75.0), 1, 1, strict=True) + \
##                   biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 740.]), (160., 0., 0.), (25.0, 75.0), (150.0, 150.0), 1, 1, strict=True)

    # create self-avoidance bias
    dendritic_bias = biases.get_elongation("sibling_repulsion", 25.0, -2) +\
                biases.get_elongation("nonrelated_repulsion", 25.0, -2)

    # somatic repulsion
    somatic_bias = biases.get_elongation("root_repulsion", 750.0, -2)
##
##    # plane boundary, push the distal apical dendrites to bend
##    plane_bias = biases.get_elongation("plane_boundary", np.array([0., 0., 900.]), (np.pi, 0.), 10, -2)
##
##    # compose the biases into the elongation bias
##    elongation_bias = [
##      (0.25, spatial_bias),
##      (0.005, dendritic_bias),
##      (0.2, somatic_bias),
##      (0.1, plane_bias)
##      ]
    # compose the biases into the elongation bias
    elongation_bias = [
      (0.25, spatial_bias),
      (0.005, dendritic_bias),
      (0.2, somatic_bias)
      ]
    
    # bifurcation biases
##    bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)
    bifurcation_bias =  biases.get_bifurcation("cross_torsion", np.pi / 3, space="ellipsoid", radii=radii)
    
##    bifurcation_internal_bias = biases.get_bifurcation("internal_branch", np.pi / 2)  

    # initialize the synthesizer for apical dendrites
##    apic_synthesizer = MorphologySynthesizer(
    basal_synthesizer = MorphologySynthesizer(
        origin=soma_position,
##        root=merge_profiles(ret['apical_dendrite']['topology'].roots),
        root=merge_profiles(ret['basal_dendrite']['topology'].roots),
        rng=misc.Random(seed),
##        theta=0,
##        phi=0,
        theta = primary_theta,
        phi=(0., 2 * np.pi),
##        axis_direction=np.array([0.0, 0.0, 1.0]),
        axis_direction=axis_direction,
        bifurcation_bias=bifurcation_bias,
##        bifurcation_internal_bias=bifurcation_internal_bias,
        elongation_bias=elongation_bias
    )

    # synthesize apical dendrites
##    apic_synthesizer.synthesize()
    basal_synthesizer.synthesize()

##    # spatial bias is a composition of truncated cones
##    spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (-1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)    
####    # synthesize basal dendrites   
####    elongation_bias = [
####      (0.005, dendritic_bias),
####      (0.2, somatic_bias)
####      ]
##    elongation_bias = [
##      (0.25, spatial_bias),
##      (0.005, dendritic_bias),
##      (0.2, somatic_bias)
##      ]    
##     
##    basal_synthesizer = MorphologySynthesizer(
##        root=merge_profiles(ret['basal_dendrite']['topology'].roots),
##        rng=misc.Random(seed),
####        theta=(np.pi / 6, np.pi * 5 / 6),
##        theta=(0, np.pi / 2),
##        phi=(0, 2 * np.pi),
##        axis_direction=np.array([0.0, 0.0, -1.0]),
##        bifurcation_bias=bifurcation_bias,
##        elongation_bias=elongation_bias,
##    )
##
##    basal_synthesizer.synthesize(soma=apic_synthesizer.soma)
##    
####    # re-use the same bias used for basal dendrites to generate oblique apical dendrites
####    # the option is_root_like make the first branch as a root
####    apic_synthesizer.synthesize(
####      is_root_like=True,
####      elongation_bias=elongation_bias,
####      )

    soma_apical = generate_apical(seed, step_size, soma_position, glom_position, axis_direction)

    for ch in soma_apical.children:
        ch.disconnect_from_parent()
        ch.connect(basal_synthesizer.soma, relation="parent")

    # return information
##    ret['apical_dendrite']['topology'] = apic_synthesizer
##    ret['apical_dendrite']['morphology'] = apic_synthesizer
    ret['basal_dendrite']['morphology'] = basal_synthesizer
    ret['output'] = basal_synthesizer.soma
    
    return ret
