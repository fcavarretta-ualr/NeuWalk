import numpy as np
from ..bias import BiasRegistry
from ..... import misc
from . import _projection
from .....core.section import Section

def _section_repulsion(reference_section, point, sections, K, n):
    directions, distances, lengths = [], [], []

    for section in sections:
        # reference point from points of a section
        points = np.array(section.points, dtype=float)

        # from current point and ghost point
        delta = point - points
        distance = np.linalg.norm(delta, axis=1)

        index0 = ~np.isclose(distance, 0.0)

        if sum(index0) < 1:
            continue

        # ignore zeros
        delta = delta[index0, :]
        distance = distance[index0]
        
        direction = (delta.T / distance).T

        points = points[index0, :]

        # step length between points
        tmp = np.linalg.norm(points[1:, :] - points[:-1, :], axis=1)

        # we anticipate what is calculate below which is now commented and thus unnecessary
        length = np.zeros(distance.size)
        length[:-1] += tmp
        length[1:]  += tmp
        length[1:-1] *= 0.5

        directions.append(direction)
        distances.append(distance)
        lengths.append(length)

    
    directions = np.vstack(directions)
    distances = np.concatenate(distances)
    lengths = np.concatenate(lengths)

    if K is None:
        factors = np.ones(directions.shape[0])
    else:
        factors = misc.hill(distances, K, n) * lengths 

    return np.sum(directions.T * factors, axis=1)

def section_repulsion(reference_section, reference_direction, sections, K, n, **kwargs):
    sections = [ section for section in sections if len(section.points) > 1 ]
    if not sections:
        return None
      
    if (K is None) != (n is None):
        raise ValueError("K and n must both be provided or both be None.")
    result = _section_repulsion(reference_section,
                        reference_section.current_point,
                        sections, K, n)
    
##    # ghost point
##    ghost_point = reference_section._generate_point(reference_direction)
##
##    
##    result0 = _section_repulsion(reference_section,
##                        reference_section.current_point,
##                        sections, K, n)
##
##    result1 = _section_repulsion(reference_section,
##                        ghost_point,
##                        sections, K, n)
##
##    # result direction
##
##    result = (result0 + result1) * 0.5

    #weight = np.linalg.norm(result)
    #result_direction = result / weight

    result = _projection.project(reference_section, result, **kwargs)    
    return result #, weight


@BiasRegistry.register_elongation("sibling_repulsion")
def sibling_repulsion(rng, random_walk, reference_direction, K, n, consider_root_like=False, **kwargs):
    sections = random_walk.siblings
    
    if consider_root_like:
        sections = [section for section in sections if section.label == random_walk.label]
        
    return section_repulsion(random_walk, reference_direction, sections, K, n, **kwargs)


@BiasRegistry.register_elongation("parent_repulsion")
def parent_repulsion(rng, random_walk, reference_direction, K, n, consider_root_like=False, **kwargs):
    sections = [random_walk.parent] if random_walk.parent else []
    
    if consider_root_like:
        sections = [section for section in sections if section.label == random_walk.label]
        
    return section_repulsion(random_walk, reference_direction, sections, K, n, consider_root_like=False, **kwargs)


@BiasRegistry.register_elongation("all_sections_repulsion")
def all_sections_repulsion(rng, random_walk, reference_direction, K, n, consider_root_like=False, **kwargs):
    sections = [
        d for d in random_walk.wholetree
        if d is not random_walk and d.label != "soma"
    ]
    
    if consider_root_like:
        sections = [section for section in sections if section.label == random_walk.label]
        
    return section_repulsion(random_walk, reference_direction, sections, K, n, consider_root_like=False, **kwargs)


@BiasRegistry.register_elongation("nonrelated_repulsion")
def nonrelated_repulsion(rng, random_walk, reference_direction, K, n, consider_root_like=False, **kwargs):
    sections = [
        d for d in random_walk.root.wholetree
        if d is not random_walk and d.label != "soma"
    ]

    parent = [random_walk.parent] if random_walk.parent else []
    siblings =  random_walk.siblings
    sections = list(set(sections) - set(parent) - set(siblings))
    
    if consider_root_like:
        sections = [section for section in sections if section.label == random_walk.label]
        
    return section_repulsion(random_walk, reference_direction, sections, K, n, **kwargs)
  



@BiasRegistry.register_elongation("root_repulsion")
def root_repulsion_bias(rng, reference_section, reference_direction, K, n, consider_root_like=False, **kwargs):
    # get the root   
    root = Section._root_and_depth(reference_section, consider_root_like=consider_root_like)['root']

    if (K is None) != (n is None):
        raise ValueError("K and n must both be provided or both be None.")


    def compute(point):
        delta = point - root.points[0]
        distance = np.linalg.norm(delta)
        
        if np.isclose(distance, 0.):
            return None
        
        direction = delta / distance
        factor = 1.0 if K is None else misc.hill(distance, K, n)     
        return direction * factor


    component1 = compute(reference_section.points[-1])

    component2 = compute(reference_section._generate_point(reference_direction))


    if component1 is None or component2 is None:
        return None
        
    return _projection.project(reference_section, (component1+component2)*0.5, **kwargs)    
