from neuwalk.analysis.morphologies import load_morphologies
import numpy as np

population = load_morphologies("/home/francesco/MorphGenPy/NeuWalk/SL", delete_labels=["unknown", "apical_oblique", "apical_secondary_oblique", "apical_secondary_dendrite", "soma", "basal_dendrite", "axon"])

cnt = []
for nrn in population:
  cnt.append(
    sum(r.bifurcation_count for r in nrn))

print(np.mean(cnt))  
