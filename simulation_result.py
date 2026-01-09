import matplotlib.pyplot as plt

# X values: percentages from 0% to 30% with 5% steps
x = [0, 5, 10, 15, 20, 25, 30]

# Three lists of success rates
base = [10, 25, 40, 55, 70, 85, 90]
additive = [5, 20, 35, 50, 65, 75, 85]
inverse = [15, 30, 45, 60, 75, 88, 95]

# Plot the lines
plt.plot(x, base, marker='o', label='Baseline')
plt.plot(x, additive, marker='s', label='Additive Score')
plt.plot(x, inverse, marker='^', label='Inverse Score')

# Labels and title
plt.xlabel('Percentage (%)')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate vs Malacious Node')

# Axis limits
plt.xlim(0, 30)
plt.ylim(0, 100)

# Grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
