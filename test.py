from neuwalk.presets.neocortex.pyramidal import generate
from neuwalk.visualization import plot_morphology

plot_morphology(generate(100)['output'])
