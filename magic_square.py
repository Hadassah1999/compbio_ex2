import random
import numpy as np
import matplotlib.pyplot as plt
import copy

P_SIZE = 0
MUTATION_RATE_IN_POPULATION = 0
MUTATION_NO_IN_INDIVIDUAL = 0
MAX_GEN = 0
ELITE_SAVED_AS_IS = 0
CROSS_OVERS_FROM_ELITE = 0

DARWIN = False
LAMARCK = False


def get_p_size():
    return P_SIZE


def get_mutation_rate_in_population():
    return MUTATION_RATE_IN_POPULATION


def get_mutation_no_in_individual():
    return MUTATION_NO_IN_INDIVIDUAL


def get_max_gen():
    return MAX_GEN


def get_elite_saved_as_is():
    return ELITE_SAVED_AS_IS


def get_cross_overs_from_elite():
    return CROSS_OVERS_FROM_ELITE


def get_darwin():
    return DARWIN


def get_lamarck():
    return LAMARCK


def set_p_size(value):
    global P_SIZE
    P_SIZE = value


def set_mutation_rate_in_population(value):
    global MUTATION_RATE_IN_POPULATION
    MUTATION_RATE_IN_POPULATION = value


def set_mutation_no_in_individual(value):
    global MUTATION_NO_IN_INDIVIDUAL
    MUTATION_NO_IN_INDIVIDUAL = value


def set_max_gen(value):
    global MAX_GEN
    MAX_GEN = value


def set_elite_saved_as_is(value):
    global ELITE_SAVED_AS_IS
    ELITE_SAVED_AS_IS = value


def set_cross_overs_from_elite(value):
    global CROSS_OVERS_FROM_ELITE
    CROSS_OVERS_FROM_ELITE = value


def set_darwin(value: bool):
    global DARWIN
    DARWIN = value


def set_lamarck(value: bool):
    global LAMARCK
    LAMARCK = value


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


def calculate_fitness(loss, avg_init_loss):
    if loss == 0:
        return 1
    else:
        return 1 / (1 + (loss / avg_init_loss))


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


def selective_mutation(square_matrix):
    attempts = 0
    best_result = copy.deepcopy(square_matrix)
    best_loss = calculate_loss(square_matrix)

    while attempts < 20:
        mutated_matrix = copy.deepcopy(square_matrix)
        mutated_matrix = mutation(mutated_matrix)
        mutated_m_loss = calculate_loss(mutated_matrix)

        if mutated_m_loss < best_loss:
            best_result = mutated_matrix
            best_loss = copy.deepcopy(mutated_m_loss)
        else:
            attempts += 1
    return best_result


def mutation(square_matrix):
    n = square_matrix.shape[0]
    
    idx1, idx2 = random.sample(range(n * n), 2)
    
    i1, j1 = divmod(idx1, n)
    i2, j2 = divmod(idx2, n)
    
    square_matrix[i1, j1], square_matrix[i2, j2] = square_matrix[i2, j2], square_matrix[i1, j1]
    
    return square_matrix


def get_new_population(prev_population, loss):
    global P_SIZE, ELITE_SAVED_AS_IS, CROSS_OVERS_FROM_ELITE
    REMAINING_POPULATION_SIZE = P_SIZE - ELITE_SAVED_AS_IS - CROSS_OVERS_FROM_ELITE

    lowest_loss_indices = np.argsort(loss)[:ELITE_SAVED_AS_IS]

    elite = [prev_population[i] for i in lowest_loss_indices]

    remaining_population_indices = np.argsort(loss)[:REMAINING_POPULATION_SIZE]
    remaining_population = [prev_population[i] for i in remaining_population_indices]

    children = []
    for i in range(0, CROSS_OVERS_FROM_ELITE - 1, 2):
        parent1, parent2 = random.sample(elite, 2)
        child1, child2 = cross_over(parent1, parent2)
        children.extend([child1, child2])

    for i in range(0, len(remaining_population) - 1, 2):
        parent1, parent2 = random.sample(remaining_population, 2)
        child1, child2 = cross_over(parent1, parent2)
        children.extend([child1, child2])

    if len(children + elite) != P_SIZE:
        for i in range(P_SIZE - (len(children) + len(elite))):
            children.append(remaining_population[-i])

    return elite + children


def calculate_next_gen_lamarckian(population, n):
    loss = np.zeros(P_SIZE)

    mutant_number_pop = int(P_SIZE * MUTATION_RATE_IN_POPULATION)

    indices = random.sample(range(P_SIZE), mutant_number_pop)

    mutant_number_ind = int(MUTATION_NO_IN_INDIVIDUAL)

    for adapted_i in range(len(population)):
        population[adapted_i] = selective_mutation((population[adapted_i]))

    for idx in indices:
        for mut_i in range(mutant_number_ind):
            population[idx] = mutation(population[idx])

    for i in range(P_SIZE):
        loss[i] = calculate_loss(population[i])

    new_population = get_new_population(population, loss)
    return new_population


def calculate_next_gen_darwinian(population, n):
    loss = np.zeros(P_SIZE)

    mutant_number_pop = int(P_SIZE * MUTATION_RATE_IN_POPULATION)

    indices = random.sample(range(P_SIZE), mutant_number_pop)

    n = population[0].shape[0]
    mutant_number_ind = int(MUTATION_NO_IN_INDIVIDUAL)

    adapted_population = copy.deepcopy(population)

    for adapted_i in range(len(adapted_population)):
        adapted_population[adapted_i] = selective_mutation((adapted_population[adapted_i]))

    for idx in indices:
        for mut_i in range(mutant_number_ind):
            population[idx] = mutation(population[idx])

    for i in range(P_SIZE):
        loss[i] = calculate_loss(adapted_population[i])

    new_population = get_new_population(population, loss)
    return new_population


def calculate_next_gen(population):
    loss = np.zeros(P_SIZE)

    mutant_number_pop = int(P_SIZE * MUTATION_RATE_IN_POPULATION)

    indices = random.sample(range(P_SIZE), mutant_number_pop)

    mutant_number_ind = int(MUTATION_NO_IN_INDIVIDUAL)

    for idx in indices:
        for mut_i in range(mutant_number_ind):
            population[idx] = mutation(population[idx])


    for i in range(P_SIZE):
        loss[i] = calculate_loss(population[i])

    new_population = get_new_population(population, loss)

    return new_population

