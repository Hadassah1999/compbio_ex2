import random
import numpy as np
import sys

P_SIZE = 100
MUTATION_RATE_IN_POPULATION = 0.4
MUTATION_RATE_IN_INDIVIDUAL = 0.03
MAX_GEN = 1000
ELITE_SAVED_AS_IS = 5
CROSS_OVERS_FROM_ELITE = 25
REMAINING_POPULATION_SIZE = P_SIZE - ELITE_SAVED_AS_IS - CROSS_OVERS_FROM_ELITE

def initialize_square(n):
    flattened_square = random.sample(range(1, n ** 2 + 1), n ** 2)
    two_d_square = np.array(flattened_square).reshape(n, n)
    return two_d_square


def loss_diagonal_pairs(square_matrix):
    n = square_matrix.shape[0]
    target = n**2 + 1
    loss = 0
    half = n // 2

    for i in range(n - half):
        a = square_matrix[i, i]
        b = square_matrix[i + half, i + half]
        loss += (a + b - target)**2

    for i in range(n - half):
        a = square_matrix[i, n - 1 - i]
        b = square_matrix[i + half, n - 1 - (i + half)]
        loss += (a + b - target)**2

    return loss

def loss_blocks(square_matrix):
    n = square_matrix.shape[0]
    target = 2 * (n**2 + 1)
    loss = 0
    for i in range(n - 1):
        for j in range(n - 1):
            block_sum = square_matrix[i,j] + square_matrix[i+1,j] + square_matrix[i,j+1] + square_matrix[i+1,j+1]
            loss += (block_sum - target)**2
    return loss

def calculate_loss(square_matrix): 

    n = square_matrix.shape[0]

    expec_sum = n * (n**2 + 1) / 2

    total_row_loss = 0

    for row_i in range(n):
        row_sum = np.sum(square_matrix[row_i])
        total_row_loss += abs(expec_sum - row_sum)

    total_column_loss = 0

    column_sums = np.sum(square_matrix, axis=0)
    column_loss_arr = np.abs(column_sums - expec_sum)
    total_column_loss = np.sum(column_loss_arr)

    first_diagonal_loss = np.abs(np.trace(square_matrix) - expec_sum)
    second_diagonal_sum = np.abs(np.trace(np.fliplr(square_matrix)) - expec_sum)

    magic_square_loss = 0

    if (n % 4 == 0):
        magic_square_loss = loss_blocks(square_matrix) + loss_diagonal_pairs(square_matrix)


    total_loss = total_row_loss + total_column_loss + first_diagonal_loss + second_diagonal_sum + magic_square_loss
    return total_loss



def to_inversion_vector(square_matrix):
    size = len(square_matrix)
    inv = [0] * size
    for i in range(size):
        inv[i] = sum(square_matrix[j] > square_matrix[i] for j in range(i))
    return inv

def crossover_inversion_vectors(inv1, inv2):
    size = len(inv1)
    point = random.randint(1, size - 2)
    child1 = inv1[:point] + inv2[point:]
    child2 = inv2[:point] + inv1[point:]
    return child1, child2

def from_inversion_vector(inv):
    size = len(inv)
    square_matrix = []
    values = list(range(1, size + 1)) 
    
    for i in reversed(range(size)):
        val = values.pop(inv[i])
        square_matrix.insert(0, val)
    
    return square_matrix



def cross_over(parent1, parent2):
    
    flat1 = parent1.flatten().tolist()
    flat2 = parent2.flatten().tolist()
    
    inv1 = to_inversion_vector(flat1)
    inv2 = to_inversion_vector(flat2)
    
    child_inv1, child_inv2 = crossover_inversion_vectors(inv1, inv2)

    child_flat1 = from_inversion_vector(child_inv1)
    child_flat2 = from_inversion_vector(child_inv2)
    
    n = parent1.shape[0]
    child1 = np.array(child_flat1).reshape(n, n)
    child2 = np.array(child_flat2).reshape(n, n)
    
    return child1, child2


def mutation(square_matrix):
    n = square_matrix.shape[0]
    
    idx1, idx2 = random.sample(range(n * n), 2)
    
    i1, j1 = divmod(idx1, n)
    i2, j2 = divmod(idx2, n)
    
    square_matrix[i1, j1], square_matrix[i2, j2] = square_matrix[i2, j2], square_matrix[i1, j1]
    
    return square_matrix

####this function is uneccessary. Edit the run algorithm function in gui.py######
def calculate_magic_square(n):
    global MUTATION_RATE_IN_POPULATION
    population = [initialize_square(n) for _ in range(P_SIZE)]
    
    best_matrix = None
    best_fitness = float('inf')
    converge = False
    gen = 0
    
    while not converge and gen < MAX_GEN:
        fitness = np.array([calculate_loss(ind) for ind in population])

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            print(best_fitness)
            best_matrix = population[min_idx]
            no_improvement = 0
        else:
            no_improvement += 1
        

        if no_improvement > 10:
                MUTATION_RATE_IN_POPULATION = min(MUTATION_RATE_IN_POPULATION * 1.5, 0.9)

        if best_fitness == 0:
            converge = True

        population = calculate_next_gen(population, n)
        gen += 1
    
    print(best_matrix)
    return best_matrix


def calculate_next_gen(population, n):
    fitness = np.zeros(P_SIZE)

    mutant_number_pop = int(P_SIZE * MUTATION_RATE_IN_POPULATION)

    indices = random.sample(range(P_SIZE), mutant_number_pop)

    mutant_number_ind = int(P_SIZE * MUTATION_RATE_IN_INDIVIDUAL)

    for idx in indices:
        for mut_i in range(mutant_number_ind):
            population[idx] = mutation(population[idx])

    for i in range(P_SIZE):
        fitness[i] = calculate_loss(population[i])
    
    lowest_loss_indices = np.argsort(fitness)[:ELITE_SAVED_AS_IS]

    elite = [population[i] for i in lowest_loss_indices]

    remaining_population_indices = np.argsort(fitness)[:REMAINING_POPULATION_SIZE]
    remaining_population = [population[i] for i in remaining_population_indices]

    children = []
    for i in range(0, CROSS_OVERS_FROM_ELITE -1, 2):
        parent1, parent2 = random.sample(elite, 2)
        child1, child2 = cross_over(parent1, parent2)
        children.extend([child1, child2])

    for i in range(0, len(remaining_population) -1, 2):
        parent1, parent2 = random.sample(remaining_population, 2)
        child1, child2 = cross_over(parent1, parent2)
        children.extend([child1, child2])

    if len(children + elite) % 2 != 0:
        children.append(remaining_population[-1])

    new_population = children + elite
    return new_population

def calculate_score(n, final_loss):
    max_loss = 0
    trials = 10000
    for _ in range(trials):
        square = initialize_square(n)
        loss = calculate_loss(square)
        if loss > max_loss:
            max_loss = loss
    loss_percantage = final_loss / max_loss * 100
    score = int(100 - loss_percantage)
    return score


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <n>")
        return
    
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Please enter a valid integer for n.")
        return

    matrix = initialize_square(n)
    print(matrix)
    
    print(calculate_magic_square(n))


if __name__ == "__main__":
    main()
