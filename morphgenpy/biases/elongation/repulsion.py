import numpy as np
from ..bias import BiasRegistry
from ... import misc
from . import _projection

def _dendrite_repulsion(reference_dendrite, point, dendrites, K, n):
    directions, distances, lengths = [], [], []

    for dendrite in dendrites:
        # reference point from points of a dendrite
        points = np.array(dendrite.points, dtype=float)

        # from current point and ghost point
        delta = point - points
        distance = np.linalg.norm(delta, axis=1)

        index0 = np.isclose(distance, 0.0)
        
        direction = (delta.T / distance).T
        if sum(index0) == direction.shape[0]:
            continue

        # ignore zeros
        direction = direction[~index0, :]
        
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
##        # let's integrate
##        directions.append(
##          (direction[:-1, :].T * weight[:-1] + direction[1:, :].T * weight[1:]) / (weight[:-1] + weight[1:]) * length)
##
##        # integral of the weights
##        weights.append((weight[:-1] + weight[1:]) * 0.5 * length)

    directions = np.vstack(directions)
    distances = np.concatenate(distances)
    lengths = np.concatenate(lengths)

    if K is None:
        factors = np.ones(directions.shape[0])
    else:
        factors = misc.hill(distances, K, n) * lengths 

    return np.sum(directions.T * factors, axis=1)

def dendrite_repulsion(reference_dendrite, reference_direction, dendrites, K, n, **kwargs):
    dendrites = [ dendrite for dendrite in dendrites if len(dendrite.points) > 1 ]
    if not dendrites:
        return None
      
    if (K is None) != (n is None):
        raise ValueError("K and n must both be provided or both be None.")

    # ghost point
    ghost_point = reference_dendrite._generate_point(reference_direction)

    
    result0 = _dendrite_repulsion(reference_dendrite,
                        reference_dendrite.current_point,
                        dendrites, K, n)

    result1 = _dendrite_repulsion(reference_dendrite,
                        ghost_point,
                        dendrites, K, n)

    # result direction
    result = (result0 + result1) * 0.5
    #weight = np.linalg.norm(result)
    #result_direction = result / weight

    result = _projection.project(reference_dendrite, result, **kwargs)    
    return result #, weight


@BiasRegistry.register_elongation("sibling_repulsion")
def sibling_repulsion(rng, random_walk, reference_direction, K, n, **kwargs):
    if random_walk.is_root_like:
        return None
    sections = random_walk.siblings
    return dendrite_repulsion(random_walk, reference_direction, sections, K, n, **kwargs)


@BiasRegistry.register_elongation("parent_repulsion")
def parent_repulsion(rng, random_walk, reference_direction, K, n, **kwargs):
    if random_walk.is_root_like:
        return None
    sections = [random_walk.parent] if random_walk.parent else []
    return dendrite_repulsion(random_walk, reference_direction, sections, K, n, **kwargs)


@BiasRegistry.register_elongation("all_dendrites_repulsion")
def all_dendrites_repulsion(rng, random_walk, reference_direction, K, n, **kwargs):
    sections = [
        d for d in random_walk.wholetree
        if d is not random_walk and d.section_type != "soma"
    ]

    return dendrite_repulsion(random_walk, reference_direction, sections, K, n, **kwargs)


@BiasRegistry.register_elongation("nonrelated_repulsion")
def nonrelated_repulsion(rng, random_walk, reference_direction, K, n, **kwargs):
    sections = [
        d for d in random_walk.absolute_root.wholetree
        if d is not random_walk and d.section_type != "soma"
    ]

    parent = [random_walk.parent] if random_walk.parent else []
    siblings =  random_walk.siblings
    sections = list(set(sections) - set(parent) - set(siblings))

    return dendrite_repulsion(random_walk, reference_direction, sections, K, n, **kwargs)
  



@BiasRegistry.register_elongation("root_repulsion")
def root_repulsion_bias(rng, reference_dendrite, reference_direction, K, n, **kwargs):
    # get the root   
    root = reference_dendrite.root
    
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


    component1 = compute(reference_dendrite.points[-1])

    component2 = compute(reference_dendrite._generate_point(reference_direction))


    if component1 is None or component2 is None:
        return None
        
    return _projection.project(reference_dendrite, (component1+component2)*0.5, **kwargs)    
