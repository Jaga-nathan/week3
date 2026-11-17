
import numpy as np
import matplotlib.pyplot as plt
import torch

# --------------------------------
# STEP 1: Activation Functions
# --------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)


# --------------------------------
# STEP 2: Loss Functions
# --------------------------------

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)

    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# --------------------------------
# STEP 3: Test Functions
# --------------------------------

print("Softmax:", np.round(
    softmax(np.array([2.0, 1.0, 0.1])), 3
))

print("ReLU:", relu(np.array([-2, 0, 3])))
print("Sigmoid:", sigmoid(np.array([0, 1, 2])))

print("MSE:", mse(
    np.array([1, 0]),
    np.array([0.8, 0.2])
))

print("Cross Entropy:", cross_entropy(
    np.array([1, 0]),
    np.array([0.8, 0.2])
))


# --------------------------------
# STEP 4: XOR Dataset
# --------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# --------------------------------
# STEP 5: Initialize Network
# --------------------------------

np.random.seed(1)

# 2 input neurons, 4 hidden neurons
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

# 4 hidden neurons, 1 output neuron
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))


# --------------------------------
# STEP 6: Training Function
# --------------------------------

def train_network(learning_rate, epochs=5000):

    global W1, b1, W2, b2

    # Reset weights for every experiment
    np.random.seed(1)

    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))

    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):

        # FORWARD PROPAGATION

        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Calculate loss

        loss = cross_entropy(y, a2)
        losses.append(loss)

        # BACKPROPAGATION

        dz2 = (a2 - y) / len(X)

        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, W2.T) * (z1 > 0)

        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # GRADIENT DESCENT

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Final predictions

    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2)

    return losses, predictions


# --------------------------------
# STEP 7: Train Network
# --------------------------------

losses, predictions = train_network(
    learning_rate=0.5,
    epochs=5000
)

print("\nXOR Predictions:")
print(np.round(predictions.flatten(), 3))


# --------------------------------
# STEP 8: Learning Rate Comparison
# --------------------------------

loss_001, _ = train_network(0.01)
loss_05, _ = train_network(0.5)
loss_5, _ = train_network(5)

# Plot loss curves

plt.figure(figsize=(10, 6))

plt.plot(loss_001, label="Learning Rate 0.01")
plt.plot(loss_05, label="Learning Rate 0.5")
plt.plot(loss_5, label="Learning Rate 5")

plt.xlabel("Epochs")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Learning Rate Comparison")

plt.legend()
plt.grid(True)
plt.savefig("xor_loss_curve.png")
plt.show()


# --------------------------------
# STEP 9: Manual Gradient Calculation
# --------------------------------

w = 1
x = 2
b = 0
t = 10

# Loss = (w*x + b - t)^2

dw = 2 * (w*x + b - t) * x
db = 2 * (w*x + b - t)

print("\nManual Gradients:")
print("dw =", dw)
print("db =", db)


# --------------------------------
# STEP 10: PyTorch Autograd
# --------------------------------

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(2.0)
t = torch.tensor(10.0)

loss = (w*x + b - t) ** 2

loss.backward()

print("\nPyTorch Autograd:")
print("w.grad =", w.grad.item())
print("b.grad =", b.grad.item())
