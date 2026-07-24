from pathlib import Path

import numpy as np

import morphgenpy.biases as biases
import morphgenpy.misc as misc
from morphgenpy.analysis.extraction import extract_statistics
from morphgenpy.analysis.morphologies import load_morphologies
from morphgenpy.misc import Random
from morphgenpy.synthesis import MorphologySynthesizer, TopologySynthesizer

from .._data import load_parameters, save_parameters
from .._preset import Preset
from .._utils import merge_profiles, merge_somas, remove_soma


DISCARDED_SECTIONS = {
    "basal_dendrite": [
        "unknown",
        "apical_oblique",
        "apical_secondary_oblique",
        "apical_secondary_dendrite",
        "soma",
        "apical_dendrite",
    ],
    "apical_dendrite": [
        "unknown",
        "apical_oblique",
        "apical_secondary_oblique",
        "apical_secondary_dendrite",
        "soma",
        "basal_dendrite",
    ],
}


def fit(
    directory,
    output,
    *,
    bin_size=10.0,
    step_size=5.0,
    n_std=1.0,
    max_attempts_per_window=10,
    max_total_attempts=1000,
):
    """Extract and save APC pyramidal generation parameters."""
    statistics = {}

    for section_type, discarded in DISCARDED_SECTIONS.items():
        params = extract_statistics(
            load_morphologies(directory, delete_section_types=discarded),
            bin_size=bin_size,
        )
        params.pop("total_length", None)
        params.pop("bifurcation_internal_density", None)
        statistics[section_type] = params

    parameters = {
        "bin_size": bin_size,
        "step_size": step_size,
        "n_std": n_std,
        "max_attempts_per_window": max_attempts_per_window,
        "max_total_attempts": max_total_attempts,
        "statistics": statistics,
    }
    return save_parameters(parameters, output)


def _build_profile(section_type, parameters, seed, verbose):
    """Generate one fitted dendritic topology."""
    synthesizer = TopologySynthesizer(
        Random(seed),
        step_size=parameters["step_size"],
        bin_size=parameters["bin_size"],
        section_type=section_type,
        **parameters["statistics"][section_type],
    )
    synthesizer.synthesize_progressive(
        n_std=parameters["n_std"],
        max_attempts_per_window=parameters["max_attempts_per_window"],
        max_total_attempts=parameters["max_total_attempts"],
        verbose=verbose,
    )
    return merge_profiles(synthesizer.roots, parameters["step_size"])


def _repulsion_bias():
    """Create a fresh self-avoidance bias."""
    return (
        biases.get_elongation("sibling_repulsion", 25.0, -2)
        + biases.get_elongation("nonrelated_repulsion", 25.0, -2)
    )


def _trajectory_parameters(apical):
    """Create fresh apical or basal trajectory parameters."""
    if apical:
        boundary_direction = (1100.0, 0.0, 0.0)
        theta = 0.0
        phi = 0.0
        axis = np.array([0.0, 0.0, 1.0])
    else:
        boundary_direction = (1100.0, np.pi, 0.0)
        theta = (0.0, np.pi / 2.0)
        phi = (0.0, 2.0 * np.pi)
        axis = np.array([0.0, 0.0, -1.0])

    boundary = biases.get_elongation(
        "truncated_cone_boundary",
        np.zeros(3),
        boundary_direction,
        (2.5, 2.5),
        (25.0, 300.0),
        1,
        1,
    )

    return {
        "theta": theta,
        "phi": phi,
        "axis_direction": axis,
        "bifurcation_bias": biases.get_bifurcation(
            "radial_torsion",
            np.pi / 3.0,
        ),
        "elongation_bias": [
            (0.5, boundary),
            (0.002, _repulsion_bias()),
        ],
        "elongation_random_weight": 5.0,
        "max_angle": np.pi / 4.0,
    }


def generate(
    parameters,
    *,
    seed=1234,
    with_soma=True,
    container=list,
    verbose=False,
):
    """Generate one APC pyramidal morphology."""
    if isinstance(parameters, (str, Path)):
        parameters = load_parameters(parameters)

    apical_profile = _build_profile(
        "apical_dendrite",
        parameters,
        seed,
        verbose,
    )
    basal_profile = _build_profile(
        "basal_dendrite",
        parameters,
        seed + 1,
        verbose,
    )

    apical = MorphologySynthesizer(
        root=apical_profile,
        rng=misc.Random(seed=seed),
        **_trajectory_parameters(apical=True),
    ).synthesize()

    basal = MorphologySynthesizer(
        root=basal_profile,
        rng=misc.Random(seed=seed + 1),
        **_trajectory_parameters(apical=False),
    ).synthesize()

    soma = merge_somas(apical, basal)
    return soma if with_soma else remove_soma(soma, container)


PRESET = Preset(
    name="apc.pyramidal",
    generate_fn=generate,
    fit_fn=fit,
    description="Anterior piriform cortex pyramidal neuron.",
)
