#implementation for regression function plot

import matplotlib.pyplot as plt

# Data
x_data = [3, -1, 7, -4, 10, 15]
y_data = [10.1, -2.5, 20.8, -20, 32, 47.2]

# OLS coefficients
beta_0 = -2.4264
beta_1 = 3.3936

# Regression function
def regression_function(x):
    return beta_0 + beta_1 * x

# Plot the data points
plt.scatter(x_data, y_data, color='blue', label='Data points')

# Plot the regression line
x_range = range(min(x_data), max(x_data) + 1)
plt.plot(x_range, [regression_function(x) for x in x_range], color='red', label='Regression line')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
