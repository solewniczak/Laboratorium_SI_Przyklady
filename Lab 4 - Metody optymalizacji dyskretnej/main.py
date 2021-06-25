import matplotlib.pyplot as plt

from helper import plot_route, plot_improvement, generate_random_coords
from algorithms import randomized_improvement, reverse, transport, permutation, simulated_annealing, genetic

nb_coords = 25
nb_experiments = 1

print('Permutation\tTransport\tReverse\tAnnealing\tGenetic')
stopping_iteration = 500
# np.random.seed(0)
# sum_delta = 0
for i in range(nb_experiments):
    coords = generate_random_coords(nb_coords)
    random_solution_permutation, _, permutation_steps = randomized_improvement(coords, permutation, stopping_iteration)
    random_solution_transport, _, transport_steps = randomized_improvement(coords, transport, stopping_iteration)
    random_solution_reverse, _, reverse_steps = randomized_improvement(coords, reverse, stopping_iteration)
    annealing_solution, _, annealing_steps = simulated_annealing(coords, stopping_iteration)
    genetic_solution, _, genetic_steps = genetic(coords, stopping_iteration)

    # delta = (random_solution_reverse - annealing_solution) / random_solution_reverse
    # sum_delta += delta

    print(f'{random_solution_permutation:.0f}', end='\t\t')
    print(f'{random_solution_transport:.0f}', end='\t\t')
    print(f'{random_solution_reverse:.0f}', end='\t\t')
    print(f'{annealing_solution:.0f}', end='\t\t')
    print(f'{genetic_solution:.0f}', end='\n')
    # print(f'{delta:.2f}', end='\n')

# print(f'Average delta: {sum_delta/nb_experiments:.3f}')

# Draw plots for last experiment
plot_improvement(coords, permutation_steps, reverse_steps, annealing_steps, genetic_steps, labels=['Permutation', 'Reverse', 'Annealing', 'Genetic'])

fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title('Permutation')
plot_route(coords, permutation_steps[-1], axs[0, 0])
axs[0, 1].set_title('Reverse')
plot_route(coords, reverse_steps[-1], axs[0, 1])
axs[1, 0].set_title('Annealing')
plot_route(coords, annealing_steps[-1], axs[1, 0])
axs[1, 1].set_title('Genetic')
plot_route(coords, genetic_steps[-1], axs[1, 1])
plt.show()