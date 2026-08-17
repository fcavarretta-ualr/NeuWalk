#from neuwalk.presets.olfactory_bulb.mitral import generate
#from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate
from neuwalk.presets.anterior_piriform_cortex.semilunar import generate
#from neuwalk.presets.neocortex.pyramidal import generate
from neuwalk.visualization import plot_morphology
m = generate(1)['output']
plot_morphology(m, section_colors={"basal_dendrite":"blue", "apical_dendrite":"red"})

print("apical_dendrite", sum(ch.total_length for ch in m.children if ch.label == "apical_dendrite"))
print("basal_dendrite", sum(ch.total_length for ch in m.children if ch.label == "basal_dendrite"))
