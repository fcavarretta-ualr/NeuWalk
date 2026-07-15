from .bias import ElongationBias, SequentialElongationBias, BifurcationBias, BiasRegistry

register_elongation = BiasRegistry.register_elongation
register_bifurcation = BiasRegistry.register_bifurcation
get_elongation = BiasRegistry.get_elongation
get_bifurcation = BiasRegistry.get_bifurcation

from . import elongation
from . import bifurcation
