from neuwalk.presets.olfactory_bulb.mitral import generate
from neuwalk.visualization import plot_morphology

plot_morphology(generate(100)['output'])
