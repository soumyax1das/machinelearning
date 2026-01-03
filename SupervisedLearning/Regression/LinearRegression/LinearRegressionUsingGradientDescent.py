import numpy as np

# 1. Sample Data (10 points)
# We will use data that shows a clear positive linear trend, plus some noise.
# x represents an independent variable, y represents the dependent variable.
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([1.5, 3.2, 4.8, 6.1, 8.5, 9.4, 11.2, 13.5, 15.0, 16.5])
N = len(X) # Number of data points

# --- Core Logic: Defining the "Best Fit" Cost Function ---

def calculate_mse(x, y, m, b):
    """
    Calculates the Mean Squared Error (MSE) for a given line (m, b) and data (x, y).
    A lower MSE means a better, more tuned line.
    """
    y_predicted = m * x + b
    # Mean Squared Error: Average of the squared errors
    mse = np.mean((y - y_predicted)**2)
    return mse

# --- Method 1: Initial Guess (Starting Point) ---

# We'll use these as the starting parameters for Gradient Descent
m_current = 0.5  # Initial guess for slope
b_current = 1.0  # Initial guess for intercept
mse_start = calculate_mse(X, Y, m_current, b_current)

print("--- Initial Poorly Tuned Line (Start of GD) ---")
print(f"Initial Slope (m): {m_current:.4f}")
print(f"Initial Intercept (b): {b_current:.4f}")
print(f"Resulting Mean Squared Error (MSE): {mse_start:.4f}")
print("-" * 50)


# --- Method 2: Iterative Solution (Gradient Descent Logic) ---

# Hyperparameters for the tuning process:
learning_rate = 0.01  # How large of a step to take down the error curve
epochs = 5000         # How many steps (iterations) to take

print(f"Starting Gradient Descent for {epochs} epochs...")

for epoch in range(epochs):
    # 1. Calculate predictions and errors
    Y_predicted = m_current * X + b_current
    # Residuals (Error): The vertical distance from the point to the line
    error = Y_predicted - Y

    # 2. Calculate the Gradients (Partial Derivatives of MSE)
    # These gradients tell us the direction and steepness of the error curve.

    # Gradient for 'm' (Slope): (2/N) * Sum(error * X)
    d_m = (2/N) * np.sum(error * X)

    # Gradient for 'b' (Intercept): (2/N) * Sum(error)
    d_b = (2/N) * np.sum(error)

    # 3. Update the Parameters (The Tuning Step)
    # Move the parameters in the *opposite* direction of the gradient, scaled by the learning rate.
    m_current = m_current - learning_rate * d_m
    b_current = b_current - learning_rate * d_b

    # Optional: Print status every 500 epochs to show progress
    if epoch % 500 == 0:
        mse_current = calculate_mse(X, Y, m_current, b_current)
        print(f"Epoch {epoch:<4}: m={m_current:.4f}, b={b_current:.4f}, MSE={mse_current:.4f}")

# --- Final Tuned Results ---

mse_final = calculate_mse(X, Y, m_current, b_current)

print("-" * 50)
print("--- Final Tuned Line (After Gradient Descent) ---")
print(f"Final Tuned Slope (m): {m_current:.3f}")
print(f"Final Tuned Intercept (b): {b_current:.3f}")
print(f"Resulting Mean Squared Error (MSE): {mse_final:.3f}")
print("-" * 50)

# Step 6: Prediction Demonstration
x_new = 12
y_predicted = m_current * x_new + b_current

print("\nPrediction Demonstration:")
print(f"The best fit line equation is: y = {m_current:.3f}x + {b_current:.3f}")
print(f"For x = {x_new}, the predicted y is: {y_predicted:.3f}")

print("\n--- Summary of Tuning Logic ---")
print("We started with a guess (MSE:",
      f"{mse_start:.3f}) and used Gradient Descent to iteratively adjust m and b.",
      f"After {epochs} steps, the MSE was minimized to ({mse_final:.3f}).")