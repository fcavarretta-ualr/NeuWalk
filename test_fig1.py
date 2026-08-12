#from neuwalk.presets.olfactory_bulb.mitral import generate
from neuwalk.presets.anterior_piriform_cortex_fig1.pyramidal import generate
#from neuwalk.presets.neocortex.pyramidal import generate
from neuwalk.visualization import plot_morphology
import matplotlib.pyplot as plt

ax = plot_morphology(generate(1234)['output'], show=False)
ax.grid(False)
ax.axis("off")

ax.set_xlim([-600, 600])
ax.set_ylim([-600, 600])
ax.set_zlim([-600, 600])
plt.show()


