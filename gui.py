import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import numpy as np
from tkinter import messagebox
from magic_square import (initialize_square, calculate_loss, calculate_next_gen, P_SIZE, MAX_GEN,
                          MUTATION_RATE_IN_POPULATION, MUTATION_RATE_IN_INDIVIDUAL, ELITE_SAVED_AS_IS,
                          CROSS_OVERS_FROM_ELITE, LAMARCK, DARWIN, calculate_next_gen_lamarckian,
                          calculate_next_gen_darwinian, calculate_score)

class MagicSquareGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magic Square Genetic Algorithm")
        self.geometry("1280x900")
        self.configure(bg="#f0f4f8")

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
        super().__init__(parent, bg="#f0f4f8")
        self.controller = controller

        label = tk.Label(self, text="Enter n for Magic Square (n x n):", font=("Helvetica", 22, "bold"), bg="#f0f4f8")
        label.pack(pady=30)

        self.entry = tk.Entry(self, textvariable=controller.n_value, font=("Helvetica", 18), width=10, justify='center')
        self.entry.pack(pady=10)

        button = tk.Button(self, text="Start", command=self.start_algorithm, font=("Helvetica", 18),
                           bg="#4CAF50", fg="white", activebackground="#45a049", padx=10, pady=5)
        button.pack(pady=20)

        algo_label = tk.Label(self, text="Select Algorithm Type:", font=("Helvetica", 18, "bold"), bg="#f0f4f8")
        algo_label.pack(pady=(30, 10))

        self.algo_var = tk.StringVar(value="Generic")
        self.algo_menu = ttk.Combobox(self, textvariable=self.algo_var, state="readonly",
                               values=["Generic", "Lamarckian", "Darwinian"], font=("Helvetica", 16))
        self.algo_menu.pack(pady=10)

        # Hyperparameters
        self.p_size = tk.IntVar(value=100)
        self.max_gen = tk.IntVar(value=1000)
        self.mutation_pop = tk.DoubleVar(value=0.4)
        self.mutation_ind = tk.DoubleVar(value=0.03)
        self.elite_count = tk.IntVar(value=5)
        self.cross_elite = tk.IntVar(value=25)

        hyper_label = tk.Label(self, text="Hyperparameters", font=("Helvetica", 18, "bold"), bg="#f0f4f8")
        hyper_label.pack(pady=(40, 10))

        self.hyper_frame = tk.Frame(self, bg="#f0f4f8")
        self.hyper_frame.pack()

        self.fields = []

        def add_hyper_field(label_text, var, row):
            label = tk.Label(self.hyper_frame, text=label_text, font=("Helvetica", 14), bg="#f0f4f8")
            entry = tk.Entry(self.hyper_frame, textvariable=var, font=("Helvetica", 14), width=15, state='disabled')
            label.grid(row=row, column=0, padx=10, pady=5, sticky="e")
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.fields.append(entry)

        add_hyper_field("Population Size:", self.p_size, 0)
        add_hyper_field("Max Generations:", self.max_gen, 1)
        add_hyper_field("Mutation Rate (Population):", self.mutation_pop, 2)
        add_hyper_field("Mutation Rate (Individual):", self.mutation_ind, 3)
        add_hyper_field("Elite Count:", self.elite_count, 4)
        add_hyper_field("Crossovers from Elite:", self.cross_elite, 5)

        self.editing = False

        def toggle_edit():
            self.editing = not self.editing
            state = 'normal' if self.editing else 'disabled'
            for entry in self.fields:
                entry.config(state=state)
            self.edit_button.config(text="Save" if self.editing else "Edit")

        self.edit_button = tk.Button(self, text="Edit", command=toggle_edit, font=("Helvetica", 14),
                                     bg="#2196F3", fg="white", activebackground="#1e88e5", padx=10)
        self.edit_button.pack(pady=10)

    def start_algorithm(self):
        n = self.controller.n_value.get()
        if n <= 2:
            messagebox.showerror("Invalid input", "n must be bigger than 2.")
            return

        global P_SIZE, MAX_GEN, MUTATION_RATE_IN_POPULATION, MUTATION_RATE_IN_INDIVIDUAL
        global ELITE_SAVED_AS_IS, CROSS_OVERS_FROM_ELITE, LAMARCK, DARWIN


        LAMARCK = False
        DARWIN = False

        selected_algo = self.algo_var.get()
        if selected_algo == "Lamarckian":
            LAMARCK = True
        elif selected_algo == "Darwinian":
            DARWIN = True

        P_SIZE = self.p_size.get()
        MAX_GEN = self.max_gen.get()
        MUTATION_RATE_IN_POPULATION = self.mutation_pop.get()
        MUTATION_RATE_IN_INDIVIDUAL = self.mutation_ind.get()
        ELITE_SAVED_AS_IS = self.elite_count.get()
        CROSS_OVERS_FROM_ELITE = self.cross_elite.get()


        self.controller.show_frame("GraphPage")
        graph_page = self.controller.frames["GraphPage"]
        graph_page.reset_page()
        threading.Thread(target=graph_page.run_algorithm).start()

class GraphPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.stop_requested = False

        # Main layout: Left = plot, Right = controls
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Plot area
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Control area (right side)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

        # Generation display inside a labeled frame
        gen_box = tk.LabelFrame(control_frame, text="Current Generation", font=("Arial", 14))
        gen_box.pack(pady=10, fill=tk.X)

        self.generation_label = tk.Label(gen_box, text="0", font=("Arial", 20), fg="blue")
        self.generation_label.pack(padx=20, pady=10)

        # Stop button (🛑)
        self.terminate_button = tk.Button(control_frame, text="Stop run 🛑", font=("Arial", 30),
                                          fg="red", command=self.terminate_run)
        self.terminate_button.pack(pady=20)

        # Status label — MOVED to control frame
        self.status_label = tk.Label(control_frame, text="Running...", font=("Arial", 16))
        self.status_label.pack(pady=10)

        # Placeholder for "View Final Matrix" button
        self.view_button = None
        self.control_frame = control_frame  # Keep reference for later

    def reset_page(self):
        self.ax.clear()
        self.canvas.draw()
        self.status_label.config(text="Running...")
        self.generation_label.config(text="0")
        self.stop_requested = False
        if self.view_button:
            self.view_button.destroy()
            self.view_button = None

    def terminate_run(self):
        self.stop_requested = True

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
        mutation_rate = MUTATION_RATE_IN_POPULATION
        fitness = np.array([calculate_loss(ind) for ind in population])

        while not converge and gen < MAX_GEN and not self.stop_requested:
            
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

            self.generation_label.config(text=str(gen))
            if no_improvement > 10:
                mutation_rate = min(mutation_rate * 1.5, 0.9)

            if no_improvement > 300:
                self.terminate_run()

            if best_fitness == 0:
                converge = True
            else:
                if LAMARCK:
                    population = calculate_next_gen_lamarckian(population, n)
                elif DARWIN:
                    population = calculate_next_gen_darwinian(population, n)
                else:
                    population = calculate_next_gen(population)

                fitness = np.array([calculate_loss(ind) for ind in population])
                if not any(np.array_equal(ind, best_matrix) for ind in population):
                    worst_idx = np.argmin(fitness)
                    population[worst_idx] = best_matrix.copy()
            
                gen += 1

        result_page = self.controller.frames["ResultPage"]
        result_page.display_matrix(best_matrix, best_fitness)
        calculate_loss(best_matrix)

        self.status_label.config(text="Run Ended. Click below to see matrix.")

        # Add button inside the control frame
        self.view_button = tk.Button(self.control_frame, text="View Final Matrix",
                                     command=lambda: self.controller.show_frame("ResultPage"),
                                     font=("Arial", 16))
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

        # Score label
        self.score_label = tk.Label(self, text="", font=("Arial", 16))
        self.score_label.pack(pady=5)

        back_button = tk.Button(self, text="Back to Start", font=("Arial", 14),
                                command=lambda: controller.show_frame("StartPage"))
        back_button.pack(pady=20)

    def display_matrix(self, matrix, final_loss):
        self.text.delete("1.0", tk.END)
        for row in matrix:
            row_str = ' '.join(f"{val:4}" for val in row)
            self.text.insert(tk.END, row_str + "\n")

        self.loss_label.config(text=f"Final Loss: {final_loss}")

        # Score calculation
        n = self.controller.n_value.get()
        score = calculate_score(n, final_loss)
        score_text = f"Score: {score}%"

        # Color code based on score
        if score < 50:
            color = "red"
        elif score < 75:
            color = "orange"
        else:
            color = "green"

        self.score_label.config(text=score_text, fg=color)


    
if __name__ == "__main__":
    app = MagicSquareGUI()
    app.mainloop()
