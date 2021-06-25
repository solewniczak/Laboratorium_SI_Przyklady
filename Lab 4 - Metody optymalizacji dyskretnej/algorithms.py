import numpy as np
from helper import fitness, plot_improvement
from genetic import Genetic


def permutation(solution):
    solution = list(solution)
    np.random.shuffle(solution)
    return solution


def transport(solution):
    solution = list(solution)
    nb_solution = len(solution)

    l = np.random.randint(2, nb_solution) # always change something
    i = np.random.randint(0, nb_solution)
    transport = solution[i: (i + l)]
    del solution[i: (i + l)]
    k = np.random.randint(0, nb_solution + 1)
    solution[k:k] = transport
    return solution


def reverse(solution):
    solution = list(solution)
    nb_solution = len(solution)

    l = np.random.randint(2, nb_solution) # always change something
    i = np.random.randint(0, nb_solution)
    solution[i: (i + l)] = reversed(solution[i: (i + l)])
    return solution


def randomized_improvement(coords, method, stopping_iteration=1000):
    nb_coords = len(coords)

    best_solution = list(range(nb_coords))
    best_fitness = fitness(coords, best_solution)

    steps = []
    for iteration in range(stopping_iteration):
        candidate_solution = method(best_solution)
        candidate_fitness = fitness(coords, candidate_solution)

        if candidate_fitness < best_fitness:
            best_fitness, best_solution = candidate_fitness, candidate_solution

        steps.append(best_solution)

    return best_fitness, best_solution, steps


def simulated_annealing(coords, stopping_iteration=1000, alpha = 0.9985):
    nb_coords = len(coords)
    T = nb_coords

    current_solution = list(range(nb_coords))
    current_fitness = fitness(coords, current_solution)

    best_solution = current_solution
    best_fitness = current_fitness

    iteration = 1
    steps = []
    while iteration < stopping_iteration:
        # choose operation
        # if np.random.randint(2) == 0:
        #     candidate_solution = transport(current_solution)
        # else:
        #     candidate_solution = reverse(current_solution)

        candidate_solution = list(current_solution)
        l = np.random.randint(2, T + 1) # lower temperature - smaller changes
        i = np.random.randint(0, nb_coords)
        candidate_solution[i: (i + l)] = reversed(candidate_solution[i: (i + l)])


        # Accept with probability 1 if candidate is better than current.
        # Accept with probability exp(-∆E/T) if candidate is worse.
        candidate_fitness = fitness(coords, candidate_solution)
        if candidate_fitness < current_fitness:
            current_fitness, current_solution = candidate_fitness, candidate_solution
            if candidate_fitness < best_fitness:
                best_fitness, best_solution = candidate_fitness, candidate_solution
        else:
            probability = np.exp(-(candidate_fitness - current_fitness) / T)
            if np.random.random() < probability:
                current_fitness, current_solution = candidate_fitness, candidate_solution

        T *= alpha
        iteration += 1
        steps.append(best_solution)

    return best_fitness, best_solution, steps


def genetic(coords, generations=500, population_size=100, elite_size=10, mutation_rate=0.01):
    genetic = Genetic(coords, population_size=population_size, elite_size=elite_size, mutation_rate=mutation_rate)

    population = genetic.initial_population()
    steps = []
    for i in range(generations):
        population = genetic.next_generation(population)
        best_solution = genetic.best_solution(population)
        steps.append(best_solution)

    best_fitness = fitness(coords, best_solution)

    return best_fitness, best_solution, steps
