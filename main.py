import random
import math
import matplotlib.pyplot as plt

# Categories of interest and their corresponding box positions
categories_of_interest = {
    'A': (0, 0),  # Top-left box
    'C': (0, 2),  # Top-right box
    'D': (1, 0),  # Middle-left box
    'E': (1, 1),  # Middle-center box
    'G': (2, 0),  # Bottom-left box
    'I': (2, 2),  # Bottom-right box
}

def init_weights(m, n):
    """Initialize weights for the Kohonen network."""
    return [[random.random() * 600.0 for _ in range(n)] for _ in range(m)]

def update_neurons(myK, x, lamb):
    """Update the weights of the neurons based on input x and learning rate lamb."""
    winner = min(range(len(myK)), key=lambda i: sum(math.pow(myK[i][j] - x[j], 2) for j in range(len(x))))
    for j in range(len(myK[0])):
        myK[winner][j] += lamb * (x[j] - myK[winner][j])
    return winner

def train_kohonen(myK, iterations, lamb_init):
    """Train the Kohonen network using competitive learning."""
    lamb = lamb_init
    delta_lamb = lamb / iterations
    used_neurons = set()  # Track used neurons

    for _ in range(iterations):
        # Randomly select a category to create input points for
        category = random.choice(list(categories_of_interest.keys()))
        row, col = categories_of_interest[category]

        # Generate a random point within the selected category's grid box
        x = col * 200 + random.random() * 200
        y = row * 200 + random.random() * 200
        input_vector = [x, y]

        # Update neurons and store the used neuron
        winner = update_neurons(myK, input_vector, lamb)
        used_neurons.add(winner)  # Mark the winning neuron as used
        lamb -= delta_lamb

    return used_neurons

def plot_initial_output(myK, num_neurons):
    """Plot the initial state of the Kohonen network."""
    plt.figure(figsize=(8, 8))
    plt.gca().set_facecolor('white')

    # Draw the grid
    for i in range(3):
        for j in range(3):
            plt.gca().add_patch(plt.Rectangle((j * 200, i * 200), 200, 200, fill=None, edgecolor='blue'))

    # Plot initial neuron positions
    for point in myK:
        plt.scatter(point[0], point[1], color='black')

    plt.xlim(0, 600)
    plt.ylim(0, 600)
    plt.title(f"Initial Output of {num_neurons} Neuron Kohonen Network")
    plt.grid()
    plt.show()

def plot_final_output(myK, used_neurons, iterations, num_neurons):
    """Plot the final state of the Kohonen network with markers for used and unused neurons."""
    plt.figure(figsize=(8, 8))
    plt.gca().set_facecolor('white')

    # Draw the grid
    for i in range(3):
        for j in range(3):
            plt.gca().add_patch(plt.Rectangle((j * 200, i * 200), 200, 200, fill=None, edgecolor='blue'))

    # Plot final neuron positions
    for i, point in enumerate(myK):
        if i in used_neurons:
            plt.text(point[0], point[1], "K", fontsize=12, ha='center', va='center', color='blue')  # Used neuron
        else:
            plt.text(point[0], point[1], "X", fontsize=12, ha='center', va='center', color='red')  # Unused neuron

    plt.xlim(0, 600)
    plt.ylim(0, 600)
    plt.title(f"Final Output of Kohonen Network after {iterations} iterations with {num_neurons} Neurons")
    plt.grid()
    plt.show()

# Parameters
num_neurons = 10  # Number of neurons to be initialized
iterations = 15000  # Total training iterations
lamb_init = 0.1  # Initial learning rate

# Initialize and plot the initial state of the Kohonen network
myK = init_weights(num_neurons, 2)
plot_initial_output(myK, num_neurons)  # Plot initial positions before training

# Train the Kohonen network
used_neurons = train_kohonen(myK, iterations, lamb_init)

# Plot the final output
plot_final_output(myK, used_neurons, iterations, num_neurons)
