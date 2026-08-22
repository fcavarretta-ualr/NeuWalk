
from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate
#from neuwalk.presets.anterior_piriform_cortex.semilunar import generate
directory = "PYR"
#directory = "SL"
from neuwalk.io import swc
import sys

seed = int(sys.argv[-1])

m = generate(seed)['output']

##cnt = 0
##for r in m.children:
##  if r.label == "apical_dendrite":
##    cnt += r.bifurcation_count
##
##with open(f"cnt{seed}.txt", "w") as fo:
##  fo.write(str(seed) + " " + str(cnt))

swc.write_swc(f"{directory}/cell{seed}.swc", [m])
