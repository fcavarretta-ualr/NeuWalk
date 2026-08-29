import json
from pathlib import Path
from ... import misc
from ...random import Random
from ...core.topology import SectionSynthesizer, connect_internal_branches, merge_trees
from ...synthesis import TopologySynthesizer, MorphologySynthesizer
from ...synthesis.morphology import biases
from .. import _common
import numpy as np

radii = np.array([2000, 1250, 1250], dtype=float)
epl_depth = 300
glom_radius = 50.

    
def generate_apical(seed, step_size, soma_position, glom_position, axis_direction):
    topol_synthesizer = TopologySynthesizer(
        Random(seed),
        step_size=step_size,
        label="apical_dendrite",
        sholl_plot={'mean':[1,1], 'std':[0,0]},
        bin_size=np.linalg.norm(soma_position-glom_position),
        primary_count_range={"min":1, "max":1}
    )

    topol_synthesizer.synthesize()


    # initializa the synthesizer for apical sections
    apical_synthesizer = MorphologySynthesizer(
        topology=topol_synthesizer.soma,
        rng=Random(seed=seed),
        theta=0,
        phi=0,
        origin=soma_position,
        axis_direction=axis_direction,
        elongation_bias=biases.get_elongation("attraction", None, None, glom_position)
    )

    apical_synthesizer.synthesize()
    
    # synthesize the tuft sections
    topol_synthesizer = TopologySynthesizer(
        Random(seed),
        step_size=step_size,
        label="apical_dendrite",
        sholl_plot={'mean':[5,10,20,40,80,80,40,20,10,5,0], 'std':[0,5,10,20,40,40,20,10,5,2.5,0]},
        bin_size=10,
        primary_count_range={"min":4, "max":6},
        with_soma=False
    )

    topol_synthesizer.synthesize()


    center = apical_synthesizer.soma.children[0].points[-1] + axis_direction * glom_radius

    # initializa the synthesizer for apical sections
    tuft_synthesizer = MorphologySynthesizer(
        topology=topol_synthesizer.roots,
        rng=Random(seed=seed),
        theta=0,
        phi=0,
        origin=apical_synthesizer.soma.children[0].points[-1],
        axis_direction=axis_direction,
        elongation_bias=biases.get_elongation("ellipsoid_boundary", np.zeros(3, dtype=float) + 50.0, 1, -1, orientation='in', center=center),
        parent=apical_synthesizer.soma.children[0]
    )

    tuft_synthesizer.synthesize()

    return apical_synthesizer

    
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
    path = Path(__file__).resolve().parent / (cell_type + ".parameters.corrected.json")
    all_params = json.loads(path.read_text())

    bin_size = kwargs.get("bin_size", 50.0)
    step_size = kwargs.get("step_size", 2.0)
    
    max_steps = kwargs.get("max_steps", 100)
    
    verbose = kwargs.get("verbose", False)
    n_std = kwargs.get("n_std", 3.0)
    max_attempts_per_window = kwargs.get("max_attempts_per_window", 100)
    max_total_attempts = kwargs.get("max_total_attempts", 2000)


    theta = kwargs.get("theta", 0.)
    phi = kwargs.get("phi", 0.)
    
    # calculate soma and glomerulus location
    soma_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, theta, phi), soma_layer)
    glom_position = misc.EllipsoidalCoordinates.to_cartesian((1 + 25 / 300.0, theta, phi), radii)
    axis_direction = misc.to_unit_vector(misc.EllipsoidalCoordinates.to_cartesian((1, theta, phi), radii))

    # generate apical and tuft dendrites
    apical_synthesizer = generate_apical(seed, step_size, soma_position, glom_position, axis_direction)


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


    # create self-avoidance bias
    section_bias = biases.get_elongation("sibling_repulsion", 20.0, -2) +\
                biases.get_elongation("nonrelated_repulsion", 10.0, -2)

    # somatic repulsion
    somatic_bias = biases.get_elongation("root_repulsion", None, None)
    
    # compose the biases into the elongation bias
    basal_elongation_bias = [
      (0.01, somatic_bias),
      (0.001, section_bias),
      (0.015, spatial_bias),
      ]
    
    # bifurcation biases
    bifurcation_bias =  biases.get_bifurcation("cross_torsion", np.pi / 3, space="ellipsoid", radii=radii)
    

    # initialize the synthesizer for apical sections
    basal_synthesizer = MorphologySynthesizer(
        origin=soma_position,
        topology=ret['basal_dendrite']['topology'].soma,
        rng=Random(seed),
        theta = primary_theta,
        phi=(0., 2 * np.pi),
        axis_direction=axis_direction,
        bifurcation_bias=bifurcation_bias,
        elongation_bias=basal_elongation_bias,
        correction_type="somatic",
        elongation_random_weight=1.0,
        elongation_bias_weight=3,
    )

    # synthesize apical sections
    basal_synthesizer.synthesize()


    # attach apical dendrite
    apical_dendrite = apical_synthesizer.soma.children[0]
    apical_dendrite.disconnect_from_parent()
    apical_dendrite.connect(basal_synthesizer.soma, relation="parent")


    # return information
    ret['output'] = basal_synthesizer.soma
    
    return ret
