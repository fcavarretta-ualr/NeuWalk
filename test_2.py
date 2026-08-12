import json
from pathlib import Path
from neuwalk.synthesis import TopologySynthesizer, MorphologySynthesizer
from neuwalk.random import Random
from neuwalk.synthesis.morphology import biases
from neuwalk.core.topology import SectionSynthesizer, connect_internal_branches, merge_trees
import numpy as np
from neuwalk.visualization import plot_morphology
import neuwalk.io.swc as swc
import sys
seed = int(sys.argv[-1])
step_size = 1
label = 'basal_dendrite'

path = Path(__file__).resolve().parent / "neuwalk/presets/anterior_piriform_cortex/semilunar.parameters.json"

params = json.loads(path.read_text())

#print(params.keys())

bc = []
ret = {}
label_select = "basal_dendrite"
for label in params.keys():
  if label != label_select: continue
  for seed in range(100):
    topol_synthesizer = TopologySynthesizer(
        Random(seed),
        step_size=step_size,
        label=label,
        **params[label]
    )

    topol_synthesizer.synthesize_progressive(        n_std=3,        max_attempts_per_window=10,        max_total_attempts=1000,        verbose=False,    )
    #topol_synthesizer.synthesize()
    #ret[label] = {'topology':topol_synthesizer}
    bc.append(
      sum(dnd.bifurcation_count for dnd in topol_synthesizer.soma.children if dnd.label == label_select))
    print(seed, bc[-1], np.mean(bc), np.std(bc))
print(np.mean(bc), np.std(bc))

### spatial bias is a composition of truncated cones
##apical_spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)
##basal_spatial_bias = biases.get_elongation("truncated_cone_boundary", np.array([0., 0., 0.]), (-1100., 0., 0.), (2.5, 2.5), (25.0, 300.0), 1, 1)
##
### create self-avoidance bias
##section_bias = biases.get_elongation("sibling_repulsion", 25.0, -2) +\
##            biases.get_elongation("nonrelated_repulsion", 25.0, -2)
##
### somatic repulsion
##somatic_bias = biases.get_elongation("root_repulsion", 750.0, -2)
##
### compose the biases into the elongation bias, one per label
##apical_elongation_bias = [
##  (0.35, apical_spatial_bias),
##  (0.005, section_bias),
##  (0.25, somatic_bias)
##  ]
##
##basal_elongation_bias = [
##  (0.35, basal_spatial_bias),
##  (0.005, section_bias),
##  (0.25, somatic_bias)
##  ]
##
### bifurcation biases
##bifurcation_bias = biases.get_bifurcation("radial_torsion", np.pi / 3)
##
### merge apical's and basal's topologies so a single
### MorphologySynthesizer can grow both labels together
##merged_topology = merge_trees(
##    ret['apical_dendrite']['topology'].soma,
##    ret['basal_dendrite']['topology'].soma,
##)
##
##merged_topology.set_order(0, labels='apical_dendrite')
##merged_topology.set_order(1, labels='basal_dendrite')
##
##synthesizer = MorphologySynthesizer(
##    topology=merged_topology,
##    rng=Random(seed),
##    theta={1:0, "default":np.pi / 3},
##    phi={1:0, "default":(0, 2 * np.pi)},
##    axis_direction={
##        'apical_dendrite': np.array([0.0, 0.0, 1.0]),
##        'basal_dendrite': np.array([0.0, 0.0, -1.0]),
##    },
##    bifurcation_bias=bifurcation_bias,
##    elongation_bias={'apical_dendrite': apical_elongation_bias, 'basal_dendrite': basal_elongation_bias}
##)
##
##
### synthesize() now runs both orders (apical, then basal) to
### completion in a single call
##synthesizer.synthesize()

##synthesizer = ret["basal_dendrite"]["topology"]
##print(sum(dnd.bifurcation_count for dnd in synthesizer.soma.children if dnd.label == "basal_dendrite"))
##for dnd in synthesizer.soma.children:
##  if dnd.label == "basal_dendrite":
##    dnd.disconnect()
##plot_morphology(synthesizer.soma)
##
##swc.write_swc('test.swc', synthesizer.soma)
##m = swc.read_swc('test.swc')
##
##print(sum(dnd.bifurcation_count for dnd in m[0].children if dnd.label == "basal_dendrite"))
##print(sum(dnd.bifurcation_count for dnd in m[0].children if dnd.label == "apical_dendrite"))
