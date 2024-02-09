#the formatted version of the lab9.ipynb which is curropted for no reason
from random import choices
import numpy as np
import matplotlib.pyplot as plt
import lab9_lib  # Assuming lab9_lib contains make_problem function

def create_problem_instance(instance_number):
    """Create a specific problem instance."""
    return lab9_lib.make_problem(instance_number)

def initialize_population(size, length):
    """Initialize a population of genomes."""
    return [choices([0, 1], k=length) for _ in range(size)]

def mutate(genome, mutation_rate):
    """Perform mutation on a genome."""
    mutation_indices = np.random.choice(len(genome), int(len(genome) * mutation_rate), replace=False)
    for i in mutation_indices:
        genome[i] = 1 - genome[i]  # Flip the bit
    return genome

def evaluate_fitness(genome, fitness_function):
    """Evaluate the fitness of a genome."""
    return fitness_function(genome)

def local_search(problem_instance_number, genome_length, population_size, mutation_rate, max_generations, convergence_threshold):
    """Local search algorithm."""
    fitness_function = create_problem_instance(problem_instance_number)
    population = initialize_population(population_size, genome_length)
    fitness_cache = {}
    fitness_history = []
    best_fitness = float('-inf')
    generations_without_improvement = 0

    for generation in range(max_generations):
        # Evaluate fitness and cache results
        fitness_values = [fitness_cache.setdefault(tuple(individual), evaluate_fitness(individual, fitness_function)) for individual in population]

        # Find the best individual in the population
        best_index = np.argmax(fitness_values)
        current_best_fitness = fitness_values[best_index]

        # Update the best fitness and check for improvement
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Termination check
        if generations_without_improvement >= convergence_threshold:
            break

        # Apply mutation to the best individual
        population[best_index] = mutate(population[best_index], mutation_rate)

        # Record fitness history
        fitness_history.append(best_fitness)

    total_fitness_calls = fitness_function.calls
    return best_fitness, total_fitness_calls, fitness_history

# Parameters
genome_length = 1000
population_size = 50  # Adjust as needed
mutation_rate = 0.15  # Advised by the professor
max_generations = 1000000  # Adjust as needed
convergence_threshold = 1000  # Adjust as needed
problem_instances = [1, 2, 5, 10]  # Adjust as needed

# Record keeping
best_fitness_per_instance = {}
fitness_calls_per_instance = {}
fitness_history_per_instance = {}

# Run the local search algorithm for each problem instance
for problem_instance_number in problem_instances:
    best_fitness, fitness_calls, fitness_history = local_search(
        problem_instance_number, genome_length, population_size, mutation_rate, max_generations, convergence_threshold
    )
    best_fitness_per_instance[problem_instance_number] = best_fitness
    fitness_calls_per_instance[problem_instance_number] = fitness_calls
    fitness_history_per_instance[problem_instance_number] = fitness_history

# Display results
for problem_instance_number in problem_instances:
    print(f"Problem Instance {problem_instance_number}: Best Fitness = {best_fitness_per_instance[problem_instance_number]}, Fitness Calls = {fitness_calls_per_instance[problem_instance_number]}")

# Visualization of fitness progression
plt.figure(figsize=(10, 6))
for problem_instance_number in problem_instances:
    fitness_history = fitness_history_per_instance[problem_instance_number]
    plt.plot(range(len(fitness_history)), fitness_history, marker='o', label=f'Instance {problem_instance_number}')
plt.title("Fitness Progression Over Generations for Different Problem Instances")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.legend()  # Display a legend to identify problem instances
plt.show()

