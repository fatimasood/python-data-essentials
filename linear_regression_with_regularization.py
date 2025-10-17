# Learning Objectives
# 1. Implement linear regression using gradient descent.
# 2. Understand how L1 (Lasso) and L2 (Ridge) regularization control overfitting.
# 3. Compare unregularized and regularized models.
# 4. Interpret how regularization affects model weights and loss over epochs.

# Import essential libraries
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Dataset (X = input, y = output)
# -------------------------------
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])   # y = 2x + 1

# -------------------------------
# Task 01: Basic Linear Regression
# -------------------------------
def linear_reg(x, y, alpha=0.01, epochs=10):
    m = np.random.randn()  # initialize slope randomly
    b = np.random.randn()  # initialize intercept randomly
    n = len(x)

    losses = []  # store loss values per epoch

    for epoch in range(epochs):
        y_pred = m * x + b
        error = y_pred - y
        loss = (1/n) * np.sum(error**2)  # MSE
        losses.append(loss)

        # Gradients
        dm = (2/n) * np.dot(error, x)
        db = (2/n) * np.sum(error)

        # Update parameters
        m -= alpha * dm
        b -= alpha * db

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, m={m:.4f}, b={b:.4f}")

    return m, b, losses

# Run the basic model
m_basic, b_basic, loss_basic = linear_reg(x, y, alpha=0.01, epochs=10)

# -------------------------------
# Task 02: Linear Regression with Regularization (L1 + L2)
# -------------------------------
def linear_reg_regularized(x, y, alpha=0.01, epochs=10, lambda_l1=0.1, lambda_l2=0.1):
    m = np.random.randn()
    b = np.random.randn()
    n = len(x)

    losses = []

    print("\nStarting Gradient Descent with L1 and L2 Regularization\n")

    for epoch in range(epochs):
        y_pred = m * x + b
        error = y_pred - y
        mse_loss = (1/n) * np.sum(error**2)

        # Gradients for MSE
        dm = (2/n) * np.dot(error, x)
        db = (2/n) * np.sum(error)

        # Regularization terms
        l1_dm = lambda_l1 * np.sign(m)
        l2_dm = 2 * lambda_l2 * m

        # Combine gradients
        dm_total = dm + l1_dm + l2_dm

        # Update parameters
        m -= alpha * dm_total
        b -= alpha * db

        # Total loss with regularization
        total_loss = mse_loss + lambda_l1 * np.abs(m) + lambda_l2 * (m**2)
        losses.append(total_loss)

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, m={m:.4f}, b={b:.4f}")

    return m, b, losses

# Run regularized model
m_reg, b_reg, loss_reg = linear_reg_regularized(x, y, alpha=0.01, epochs=10, lambda_l1=0.1, lambda_l2=0.1)

# -------------------------------
# Task 03: Compare Results (Plots)
# -------------------------------
plt.figure(figsize=(12, 4))

# Loss vs Epochs
plt.subplot(1, 3, 1)
plt.plot(loss_basic, label='Without Regularization')
plt.plot(loss_reg, label='With Regularization')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Weight (m) Comparison
plt.subplot(1, 3, 2)
plt.bar(['Basic', 'Regularized'], [m_basic, m_reg], color=['blue', 'red'])
plt.title('Slope (m) Comparison')
plt.ylabel('Weight Value')
plt.grid(True, axis='y')

# Predicted Line vs True Line
plt.subplot(1, 3, 3)
plt.scatter(x, y, color='black', label='True Data')
plt.plot(x, m_basic * x + b_basic, color='blue', label='Basic Fit')
plt.plot(x, m_reg * x + b_reg, color='red', linestyle='--', label='Regularized Fit')
plt.title('Predicted Line vs True Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Task 04: Elastic Net Regularization
# -------------------------------
def linear_reg_elastic_net(x, y, lambda_l1=0.1, lambda_l2=0.1, epochs=10, alpha=0.01):
    m = np.random.randn()
    b = np.random.randn()
    n = len(x)

    losses = []

    print("\nStarting Gradient Descent with Elastic Net Regularization\n")

    for epoch in range(epochs):
        y_pred = m * x + b
        error = y_pred - y
        mse_loss = (1/n) * np.sum(error**2)

        # Gradients
        dm = (2/n) * np.dot(error, x)
        db = (2/n) * np.sum(error)

        # Elastic Net Regularization
        l1_dm = lambda_l1 * np.sign(m)
        l2_dm = 2 * lambda_l2 * m
        dm_total = dm + l1_dm + l2_dm

        # Update parameters
        m -= alpha * dm_total
        b -= alpha * db

        total_loss = mse_loss + lambda_l1 * np.abs(m) + lambda_l2 * (m**2)
        losses.append(total_loss)

        print(f"Epoch {epoch+1}: Total Loss={total_loss:.4f}, m={m:.4f}, b={b:.4f}")

    return m, b, losses

# Run Elastic Net model
m_enet, b_enet, loss_enet = linear_reg_elastic_net(x, y, lambda_l1=0.1, lambda_l2=0.1, epochs=10, alpha=0.01)


# Task: Experiment with Parameters

print ( "\nExperimenting with different learning rates and regularization L1, L2:\n")
print ("For Learning Rate: 0.001, L1: 0.01, L2: 0.01")
m_exp, b_exp = linear_reg_regularized(x, y, alpha=0.001, epochs=3, lambda_l1=0.01, lambda_l2=0.01)
print ("\nFor Learning Rate: 0.01, L1: 0.5, L2: 0.5")
m_exp2, b_exp2 = linear_reg_elastic_net(x, y, alpha=0.01, epochs=3, lambda_l1=0.5, lambda_l2=0.5)
print ("\nFor Learning Rate: 0.1, L1: 0.1, L2: 0.1")
m_exp3, b_exp3 = linear_reg(x, y, alpha=0.1, epochs=3, lambda_l1=0.1, lambda_l2=0.1) 