#from neuwalk.presets.anterior_piriform_cortex.semilunar import generate
#from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate
from neuwalk.presets.neocortex.pyramidal import generate
#from neuwalk.presets.olfactory_bulb.mitral import generate
from neuwalk.visualization import plot_morphology


plot_morphology(generate(0)['output'])
