import random
import numpy as np

def initialize_square(n):
    flattened_square = random.sample(range(1, n ** 2 + 1), n ** 2)
    two_d_square = np.array(flattened_square).reshape(n, n)
    return two_d_square


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

    total_loss = total_row_loss + total_column_loss + first_diagonal_loss + second_diagonal_sum
    return total_loss

