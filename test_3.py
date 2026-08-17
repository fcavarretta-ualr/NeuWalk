#from neuwalk.presets.olfactory_bulb.mitral import generate
from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate
#from neuwalk.presets.anterior_piriform_cortex.semilunar import generate
#from neuwalk.presets.neocortex.pyramidal import generate
from neuwalk.presets._common import synthesize_topologies
import numpy as np
import json
from pathlib import Path

step_size = 0.5 #2.5
n_std = 3
path = Path("neuwalk/presets/anterior_piriform_cortex/pyramidal.parameters.json")
all_params = json.loads(path.read_text())
del all_params['basal_dendrite']
    
cnt = []
for seed in range(200):
  ret = synthesize_topologies(all_params, seed, step_size, n_std, 10, 1000, False, with_soma=None)['apical_dendrite']
  cnt.append(ret['topology'].soma.bifurcation_count)
  print(seed, cnt[-1], np.mean(cnt), np.std(cnt))


#soma = generate(seed, step_size=step_size)['output']
#print(ret['topology'].soma.bifurcation_count, sum(s.bifurcation_count for s in soma.children if s.label == "apical_dendrite"))
#print(np.mean(cnt), np.std(cnt))
