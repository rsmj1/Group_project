import numpy as np

def partially_mapped_crossover(parent1, parent2):
    """
    Perform Partially Mapped Crossover (PMX) on two individuals.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: A new individual created through PMX.
    """
    candidate1 = list(parent1.route)
    candidate2 = list(parent2.route)
    if len(candidate1) != len(candidate2):
        print('Candidate solutions must have the same length')
        return 0

    length = len(candidate1)
    index_set = set()
    full_set = set(range(length))

    # Choose our crossover points:
    my_range = np.random.choice(length, 2, False)
    a = min(my_range)
    b = max(my_range)

    for j in range(a, b):
        index_set.add(j)

    # Initialize an empty offspring:
    offspring = [None] * (length - 1)

    if a > b:
        middle_section1 = candidate1[b:a + 1]
        middle_section2 = candidate2[b:a + 1]

        offspring[b:a] = middle_section1
        for j in range(b, a + 1):
            index_set.add(j)
    elif b > a:
        middle_section1 = candidate1[a:b + 1]
        middle_section2 = candidate2[a:b + 1]

        offspring[a:b] = middle_section1
        for j in range(a, b):
            index_set.add(j + 1)

    for count, item in enumerate(middle_section2):
        if item not in set(middle_section1):
            item2 = middle_section1[count]
            index = candidate2.index(item2)
            recursive_fill(index, index_set, item, offspring, candidate2)

    for final_item in full_set ^ index_set:
        offspring[final_item] = candidate2[final_item]

    return Individual(route=np.array(offspring), alpha=combine_alphas(parent1.alpha, parent2.alpha))

def recursive_fill(index, index_set, item, offspring, candidate):
    if index not in index_set:
        offspring[index] = item
        index_set.add(index)
    else:
        new_item = offspring[index]
        new_index = candidate.index(new_item)
        recursive_fill(new_index, index_set, item, offspring, candidate)


def edge_crossover(parent1, parent2):
    """
    Perform Edge Crossover on two individuals.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: A new individual created through Edge Crossover.
    """
    candidate1 = list(parent1.route)
    candidate2 = list(parent2.route)
    if len(candidate1) != len(candidate2):
        print('Candidate solutions must have the same length')
        return 0

    length = len(candidate1)
    neighbor_lists = {}

    for node in range(length):
        neighbor_lists[node] = set([candidate1[(node + 1) % length], candidate1[(node - 1) % length]])

    child = [None] * length
    current_node = candidate1[0]

    for i in range(length):
        child[i] = current_node
        for neighbor in neighbor_lists:
            neighbor_lists[neighbor].discard(current_node)
        next_nodes = list(neighbor_lists[current_node])

        if next_nodes:
            current_node = np.random.choice(next_nodes)
        else:
            remaining_nodes = set(candidate1) - set(child)
            if remaining_nodes:
                current_node = np.random.choice(list(remaining_nodes))

    return Individual(route=np.array(child), alpha=combineAlphas(parent1.alpha, parent2.alpha))