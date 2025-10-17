#Learning Objectives

#1. Implement linear regression using gradient descent.
#2. Understand how L1 (Lasso) and L2 (Ridge) regularization control overfitting.
#3. Compare unregularized and regularized models.
#4. Interpret how regularization affects model weights and loss over epochs.


#imoort essential librarries numpy for numerical operations and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt

# Dataset (X = input, y = output)
x = np.array([1, 2, 3, 4, 5])
y = np.array([3,5,7,9,11])


# task # 01: Implement Basic Linear Regression

def liner_reg(X,Y, alpha = 0.01, epochs = 10):  #create function
    m = np.random.randn()  #initialize slope with random values
    b = np.random.randn()  #initialize intercept with random values
    n = len(X)  #number of data points

    for epoch in range(epochs):  #loop through epochs
        y_pred = m*X+b #predicted values
        error = y_pred - Y  #difference

        # Gradients
        dm =(2/n)* np.dot(error, X)  #gradient for slope
        db = (2/n)* np.sum(error)  #gradient for intercept

        # Update parameters
        m = m - alpha * dm
        b = b - alpha * db

        print(f"Epoch {epoch+1}: , slope={m:.4f}, intecept={b:.4f}")

    return m, b #return slope and intercept

# Run the function
m, b = liner_reg(x, y,  alpha = 0.01, epochs = 10)
