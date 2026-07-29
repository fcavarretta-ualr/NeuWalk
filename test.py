from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate
from neuwalk.visualization import plot_morphology

plot_morphology(generate(100)['output'])
