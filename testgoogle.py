import numpy as np
import matplotlib.pyplot as plt

# Example parameters for 10 levels
parameters = [
    {'a': -1.0, 'b': 0.4, 'c': 2.0},
    {'a': -1.2, 'b': 0.5, 'c': 2.1},
    {'a': -0.9, 'b': 0.6, 'c': 1.8},
    # Add more parameter sets for 10 levels...
]

# Calculate half-life (t_{1/2}) for each level
half_lives = [np.log(2) / p['b'] for p in parameters]

# Plotting
levels = np.arange(1, len(parameters) + 1)
plt.plot(levels, half_lives, marker='o', linestyle='-')
plt.xlabel('Levels')
plt.ylabel('Half-life (t_{1/2})')
plt.title('Comparison of Growth Characteristics Across Levels')
plt.grid(True)
plt.show()
