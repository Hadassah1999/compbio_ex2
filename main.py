import random
import numpy as np
import pygame
import sys

P_SIZE = 100
MUTATION_RATE = 0.1
MAX_GEN = 200



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

def draw_matrix_on_surface(matrix, surface, cell_size, font, margin=50):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            rect = pygame.Rect(margin + j*cell_size, margin + i*cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, (200, 200, 200), rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 1)
            
            text = font.render(str(val), True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            surface.blit(text, text_rect)

def calculate_magic_square(n):
    population = np.zeros(P_SIZE)

    for i in range(P_SIZE):
        population[i] = initialize_square(n)


    converge = False
    i = 0
    while(not converge or i < MAX_GEN):
        fitness = np.zeros(P_SIZE)
        for i in range(P_SIZE):
            fitness[i] = calculate_loss(population[i])
            if (fitness[i] == 0):
                converge = True
        
        population = calculate_next_gen(population, n)
        i += 1


    



def calculate_next_gen(population, n):

    fitness = np.zeros(P_SIZE)

    mutant_number = int(n * MUTATION_RATE)

    indices = random.sample(range(n), mutant_number) 

    for idx in indices:
        population[idx] = mutation(population[idx])

    for i in range(P_SIZE):
        fitness[i] = calculate_loss(population[i])
    
    highest_fitness = np.argsort(fitness)[:2] 

    elite = [population[i] for i in highest_fitness]

    remaining_indices = [i for i in range(P_SIZE) if i not in highest_fitness]
    remaining_population = [population[i] for i in remaining_indices]
    
    children = np.array
    for i in range(0, len(remaining_population) -1, 2):
        parent1 = remaining_population[i]
        parent2 = remaining_population[i + 1]
        child1, child2 = cross_over(parent1, parent2)
        children.extend([child1, child2])

        if len(remaining_population) % 2 != 0:
            children.append(remaining_population[-1])

    new_population = children[:P_SIZE - 2] + elite
    return new_population


def to_inversion_vector(square_matrix):
    n = len(square_matrix)
    inv = [0] * n
    for i in range(n):
        inv[i] = sum(1 for j in range(i) if square_matrix[j] > square_matrix[i])
    return inv

def crossover_inversion_vectors(inv1, inv2):
    n = len(inv1)
    point = random.randint(1, n - 1)
    child1 = inv1[:point] + inv2[point:]
    child2 = inv2[:point] + inv1[point:]
    return child1, child2

def from_inversion_vector(inv):
    n = len(inv)
    square_matrix = []
    values = list(range(1, n + 1)) 
    
    for i in reversed(range(n)):
        val = values.pop(inv[i])
        square_matrix.insert(0, val)
    
    return square_matrix


def mutation(square_matrix):
    n = square_matrix.shape[0]
    
    idx1, idx2 = random.sample(range(n * n), 2)
    
    i1, j1 = divmod(idx1, n)
    i2, j2 = divmod(idx2, n)
    
    square_matrix[i1, j1], square_matrix[i2, j2] = square_matrix[i2, j2], square_matrix[i1, j1]
    
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



def main():
    pygame.init()
    
    # Fixed screen size
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 600
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("Magic Square GUI")

    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    BLACK = (0, 0, 0)
    BLUE = (50, 150, 255)

    input_font = pygame.font.Font(None, 48)
    instruction_font = pygame.font.Font(None, 36)

    input_box = pygame.Rect(200, 150, 200, 50)
    button_box = pygame.Rect(250, 220, 100, 40)
    user_text = ''
    active = False

    current_screen = "input"
    matrix = None
    matrix_surface = None
    scroll_x, scroll_y = 0, 0
    cell_size = 60
    margin = 50
    font = None

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if current_screen == "input":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    active = input_box.collidepoint(event.pos)
                    if button_box.collidepoint(event.pos):
                        try:
                            n = int(user_text)
                            if n > 0:
                                matrix = initialize_square(n)

                                # Create large surface for matrix
                                surf_width = n * cell_size + 2 * margin
                                surf_height = n * cell_size + 2 * margin
                                matrix_surface = pygame.Surface((surf_width, surf_height))
                                font_size = int(cell_size * 0.5)
                                font = pygame.font.Font(None, font_size)
                                draw_matrix_on_surface(matrix, matrix_surface, cell_size, font, margin)

                                scroll_x, scroll_y = 0, 0
                                current_screen = "matrix"
                        except ValueError:
                            user_text = ''
                elif event.type == pygame.KEYDOWN and active:
                    if event.key == pygame.K_RETURN:
                        try:
                            n = int(user_text)
                            if n > 0:
                                matrix = initialize_square(n)

                                surf_width = n * cell_size + 2 * margin
                                surf_height = n * cell_size + 2 * margin
                                matrix_surface = pygame.Surface((surf_width, surf_height))
                                font_size = int(cell_size * 0.5)
                                font = pygame.font.Font(None, font_size)
                                draw_matrix_on_surface(matrix, matrix_surface, cell_size, font, margin)

                                scroll_x, scroll_y = 0, 0
                                current_screen = "matrix"
                        except ValueError:
                            user_text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode

            elif current_screen == "matrix":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        scroll_y = min(scroll_y + 40, 0)
                    elif event.key == pygame.K_DOWN:
                        scroll_y -= 40
                    elif event.key == pygame.K_LEFT:
                        scroll_x = min(scroll_x + 40, 0)
                    elif event.key == pygame.K_RIGHT:
                        scroll_x -= 40

        screen.fill(WHITE)

        if current_screen == "input":
            pygame.draw.rect(screen, BLUE if active else GRAY, input_box, 2)
            txt_surface = input_font.render(user_text, True, BLACK)
            screen.blit(txt_surface, (input_box.x + 10, input_box.y + 10))

            instruction = instruction_font.render("Enter size n for the magic square:", True, BLACK)
            screen.blit(instruction, (120, 100))
            pygame.draw.rect(screen, BLUE, button_box)
            button_text = instruction_font.render("Submit", True, WHITE)
            screen.blit(button_text, (button_box.x + 10, button_box.y + 5))

        elif current_screen == "matrix":
            if matrix_surface:
                screen.blit(matrix_surface, (scroll_x, scroll_y))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

