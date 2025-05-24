import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import numpy as np
from magic_square import initialize_square, calculate_loss, calculate_next_gen, P_SIZE, MAX_GEN, MUTATION_RATE

class MagicSquareGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magic Square Genetic Algorithm")
        self.geometry("1000x700")

        self.n_value = tk.IntVar()
        self.frames = {}

        for F in (StartPage, GraphPage, ResultPage):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Enter n for Magic Square (n x n):", font=("Arial", 18))
        label.pack(pady=30)

        self.entry = tk.Entry(self, textvariable=controller.n_value, font=("Arial", 16))
        self.entry.pack(pady=10)

        button = tk.Button(self, text="Start", command=self.start_algorithm, font=("Arial", 16))
        button.pack(pady=20)

    def start_algorithm(self):
        self.controller.show_frame("GraphPage")
        graph_page = self.controller.frames["GraphPage"]
        graph_page.reset_page()
        threading.Thread(target=graph_page.run_algorithm).start()

class GraphPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Min Loss per Generation", font=("Arial", 18))
        label.pack(pady=10)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(self, text="Running...", font=("Arial", 16))
        self.status_label.pack(pady=10)

        self.view_button = None

    def reset_page(self):
        self.ax.clear()
        self.canvas.draw()
        self.status_label.config(text="Running...")
        if self.view_button:
            self.view_button.destroy()
            self.view_button = None

    def run_algorithm(self):
        n = self.controller.n_value.get()
        population = [initialize_square(n) for _ in range(P_SIZE)]

        best_matrix = None
        best_fitness = float('inf')
        loss_over_gens = []
        generations = []

        converge = False
        gen = 0
        no_improvement = 0
        mutation_rate = MUTATION_RATE

        while not converge and gen < MAX_GEN:
            fitness = np.array([calculate_loss(ind) for ind in population])
            min_idx = np.argmin(fitness)

            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_matrix = population[min_idx]
                generations.append(gen)
                loss_over_gens.append(best_fitness)
                self.update_plot(generations, loss_over_gens)
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > 10:
                mutation_rate = min(mutation_rate * 1.5, 0.9)

            if best_fitness == 0:
                converge = True

            population = calculate_next_gen(population, n)
            gen += 1

        result_page = self.controller.frames["ResultPage"]
        result_page.display_matrix(best_matrix, best_fitness)
        self.status_label.config(text="Run Ended. Click below to see matrix.")

        self.view_button = tk.Button(self, text="View Final Matrix", command=lambda: self.controller.show_frame("ResultPage"), font=("Arial", 16))
        self.view_button.pack(pady=10)

    def update_plot(self, gens, losses):
        self.ax.clear()
        self.ax.set_title("Improved Loss Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Loss")
        self.ax.plot(gens, losses, 'bo-')
        self.canvas.draw()

class ResultPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Final Magic Square", font=("Arial", 18))
        label.pack(pady=20)

        self.text = tk.Text(self, font=("Courier", 14), width=60, height=20)
        self.text.pack(pady=10)

        self.loss_label = tk.Label(self, text="", font=("Arial", 16))
        self.loss_label.pack(pady=10)

        back_button = tk.Button(self, text="Back to Start", font=("Arial", 14),
                                command=lambda: controller.show_frame("StartPage"))
        back_button.pack(pady=20)

    def display_matrix(self, matrix, final_loss):
        self.text.delete("1.0", tk.END)
        for row in matrix:
            row_str = ' '.join(f"{val:4}" for val in row)
            self.text.insert(tk.END, row_str + "\n")

        self.loss_label.config(text=f"Final Loss: {final_loss}")

if __name__ == "__main__":
    app = MagicSquareGUI()
    app.mainloop()
