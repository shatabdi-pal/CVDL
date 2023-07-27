#implementation for gradient descent
import numpy as np

def loss_function(w):
    return w[0] + 2 + 4*w[1] - 4 + 2*w[2] -2*w[3] + ( -2*w[2] + 2*w[3])

def gradient(w):
    gradient_w1 = w[0] + 2
    gradient_w2 = 4*w[1] - 4
    gradient_w3 = 2*w[2] -2*w[3]
    gradient_w4 = -2*w[2] + 2*w[3]
    return np.array([gradient_w1, gradient_w2, gradient_w3, gradient_w4])

def gradient_descent(eta, num_iterations):
    w = np.array([0, 0, 0, 0])

    for _ in range(num_iterations):
        grad = gradient(w)
        w = w - eta * grad

    return w

if __name__ == "__main__":
    # Set hyperparameters
    learning_rate = 0.1
    num_iterations = 1000

    # Run gradient descent to find the minimum value of L(w)
    w_min = gradient_descent(learning_rate, num_iterations)
    min_loss = loss_function(w_min)

    print("Minimum value of L(w):", min_loss)
    print("Optimal w:", w_min)
