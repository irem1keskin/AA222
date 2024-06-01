import numpy as np
import random

# Define airfoil types and their properties (simplified example)
airfoils = {
    'Cyl': {'Cl': 0.1, 'Cd': 0.05, 'thickness': 0.2},
    'DU21': {'Cl': 0.948, 'Cd': 0.066, 'thickness': 0.18},
    'DU25': {'Cl': 1.062, 'Cd': 0.079, 'thickness': 0.25},
    'DU30': {'Cl': 1.256, 'Cd': 0.112, 'thickness': 0.3},
    'DU35': {'Cl': 1.26, 'Cd': 0.12, 'thickness': 0.35},
    'DU40': {'Cl': 0.967, 'Cd': 0.135, 'thickness': 0.4},
    'NACA64': {'Cl': 1.011, 'Cd': 0.058, 'thickness': 0.18}
}

desired_order = ['Cyl', 'DU40', 'DU35', 'DU30', 'DU25', 'DU21', 'NACA64']
desired_order_indices = {af: i for i, af in enumerate(desired_order)}

population_size = 1000
generations = 200
crossover_rate = 0.7
mutation_rate = 0.1
weights = {'efficiency': 0.7, 'structural': 0.1, 'smoothness': 0.1, 'manufacturability': 0.05, 'variety': 0.05}

def generate_initial_population(size):
    population = []
    for _ in range(size):
        individual = [random.choice(desired_order[:min(i//2 + 1, len(desired_order))]) for i in range(17)]
        population.append(individual)
    return population

#fitness
def calculate_fitness(individual):
    efficiency = np.mean([airfoils[af]['Cl'] / airfoils[af]['Cd'] for af in individual])
    structural = np.mean([airfoils[af]['thickness'] for af in individual[:6]])
    smoothness = np.mean([abs(airfoils[individual[i]]['thickness'] - airfoils[individual[i+1]]['thickness']) for i in range(16)])
    manufacturability = 1 / (np.std([airfoils[af]['thickness'] for af in individual]) + 1e-6) 

    order_penalty = sum([1 for i in range(16) if desired_order_indices[individual[i]] > desired_order_indices[individual[i+1]]])
    
    unique_airfoils = len(set(individual))
    variety_incentive = unique_airfoils / len(individual)
    
    fitness = (weights['efficiency'] * efficiency + 
               weights['structural'] * structural + 
               weights['smoothness'] * smoothness +
               weights['manufacturability'] * manufacturability + 
               weights['variety'] * variety_incentive - 
               order_penalty)
    return fitness 


def select_parents(population, fitnesses):
    parents = random.choices(population, weights=fitnesses, k=2)
    return parents


def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, 15)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        # Ensure offspring adhere to desired order
        offspring1 = enforce_order(offspring1)
        offspring2 = enforce_order(offspring2)
        return offspring1, offspring2
    return parent1, parent2


def enforce_order(individual):
    for i in range(1, len(individual)):
        if desired_order_indices[individual[i]] < desired_order_indices[individual[i-1]]:
            individual[i] = random.choice(desired_order[:desired_order_indices[individual[i-1]]+1])
    return individual


def mutate(individual):
    if random.random() < mutation_rate:
        point = random.randint(0, 16)
        valid_choices = desired_order[:min(point//3 + 1, len(desired_order))]
        individual[point] = random.choice(valid_choices)
    return enforce_order(individual)


def genetic_algorithm():
    population = generate_initial_population(population_size)
    for _ in range(generations):
        fitnesses = [calculate_fitness(ind) for ind in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        population = new_population
    best_individual = max(population, key=calculate_fitness)
    return best_individual

optimal_airfoil_distribution = genetic_algorithm()
print("Optimal Airfoil Distribution:", optimal_airfoil_distribution)
