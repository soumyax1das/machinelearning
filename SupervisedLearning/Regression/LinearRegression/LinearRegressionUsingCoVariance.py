"""
This code is to understand how to implement Linear Regression without using any machine learning library.
This way we can understand how iteratively the best fit line is found.
"""

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
    MSE is the common cost function used to determine how "bad" the fit is.
    A lower MSE means a better, more tuned line.
    """
    y_predicted = m * x + b
    # Squared Error: (Actual Y - Predicted Y)^2
    squared_errors = (y - y_predicted)**2
    print("\nPredicted Y values: " + str(y_predicted))
    print("Actual Y values: " + str(y))
    print("Squared errors" + str(squared_errors))
    # Mean Squared Error: Average of the squared errors
    mse = np.mean(squared_errors)
    print("MSE: " + str(mse))
    return mse

# --- Method 1: Initial Guess (Poorly Tuned Line) ---
if __name__ == '__main__':
    print("\n--- Linear Regression Using Basic API ---\n")
    print("Trying to define a best fit line using basic calculations without ML libraries.\n")
    print("y = mx + b")


    # Initial guess for slope (m) and intercept (b) - need a starting point
    m_guess = 0.5  # Initial guess for slope
    b_guess = 1.0  # Initial guess for intercept
    mse_guess = calculate_mse(X, Y, m_guess, b_guess)

    print("-" * 50)
    print("--- Initial Poorly Tuned Line (Guess) ---")
    print(f"Initial Slope (m): {m_guess:.3f}")
    print(f"Initial Intercept (b): {b_guess:.3f}")
    print(f"Resulting Mean Squared Error (MSE): {mse_guess:.3f}")
    print("-" * 50)



# --- Method 2: Analytical Solution (The Least Squares Logic) ---
# This is the mathematical 'tuning' that finds the perfect m and b directly.

# Step 1 & 2: Calculate the mean of X and Y
x_mean = np.mean(X)
print(X)
print(x_mean.round(2))
y_mean = np.mean(Y)
print(Y)
print(y_mean.round(2))


# Step 3: Calculate the optimal slope (m)
# Formula: m = Sum((xi - x_mean) * (yi - y_mean)) / Sum((xi - x_mean)^2)

print(X - x_mean)
print(Y - y_mean)
print((X - x_mean) * (Y - y_mean))

# Calculate the sums needed for the slope formula, here we find {covariance(X,y) / variance of X}
# That is the best slope here
"""
That's a fantastic question. To understand why that ratio gives the "best" slope ($m_{\text{best}}$), we need to look at what the numerator and denominator are actually measuring.This formula comes from the algebraic solution to minimizing the Mean Squared Error (MSE), but you can understand the intuition by relating it to two core statistical concepts: Covariance and Variance.1. The Numerator: $\sum (X - \bar{X})(Y - \bar{Y})$ (Covariance)The numerator measures the joint variability or the degree to which $X$ and $Y$ change together.

- If $X$ increases and $Y$ tends to increase as well, this sum will be positive, indicating a positive relationship.
- If $X$ increases and $Y$ tends to decrease, the sum will be negative, indicating a negative relationship.
- If there's no consistent relationship, the sum will be close to zero.
- This is essentially the covariance between $X$ and $Y$, which tells us how much $Y$ changes for a unit change in $X$ on average.
2. The Denominator: $\sum (X - \bar{X})^2$ (Variance of X)The denominator measures the variability of $X$ itself.
- It quantifies how spread out the values of $X$ are around their mean.
- A larger variance means that $X$ values are more spread out, while a smaller variance means they are closer to the mean.
- This is the variance of $X$, which normalizes the covariance by the spread of $X$ values.
Putting It Together: The RatioThe slope $m_{\text{best}}$ is the ratio of these two quantities:

The term $(x_i - \bar{x})$ tells you how far a point $x_i$ is from the average $\bar{x}$.The term $(y_i - \bar{y})$ tells you how far a point $y_i$ is from the average $\bar{y}$.Positive Contribution (Positive Slope): If an $x_i$ is above its mean ($\bar{x}$) AND its corresponding $y_i$ is also above its mean ($\bar{y}$), the product is positive $(+ \times +)$. Similarly, if both are below their respective means $(- \times -)$, the product is still positive. A large, positive sum means $X$ and $Y$ generally increase together, indicating a strong positive slope.Negative Contribution (Negative Slope): If $x_i$ is above $\bar{x}$ but $y_i$ is below $\bar{y}$, the product is negative $(+ \times -)$. A large, negative sum means as $X$ increases, $Y$ decreases, indicating a strong negative slope.Conclusion for Numerator: It determines the direction and strength of the relationship.2. The Denominator: $\sum (X - \bar{X})^2$ (Variance of X)The denominator measures the spread or total variability of the $X$ values alone.The term $(x_i - \bar{x})^2$ is simply the squared distance of each $X$ point from the mean of $X$.Conclusion for Denominator: It acts as a normalizing factor.3. Why the Ratio is the "Best" SlopeThe slope ($m$) represents the rate of change: "For every one-unit change in $X$, how much does $Y$ change?"The optimal slope $m_{\text{best}}$ is essentially defined as:$$m = \frac{\text{Change in Y due to X}}{\text{Change in X}}$$By taking the ratio:$$\text{Slope } (m) = \frac{\text{Covariance of } (X, Y)}{\text{Variance of } (X)}$$You are calculating the relationship between $X$ and $Y$ (the numerator) and scaling it down by the total spread of $X$ (the denominator). This scaling is crucial because it gives you the correct rate that minimizes the vertical distances (residuals) between the data points and the line.In short, this formula calculates the ideal rate of change ($m$) that perfectly balances the spread of your $X$ data against the joint movement of $X$ and $Y$. It is the only slope value that ensures the errors on both sides of the line are perfectly minimized according to the squared error metric.
"""


numerator_sum = np.sum((X - x_mean) * (Y - y_mean))
denominator_sum = np.sum((X - x_mean)**2)
m_best = numerator_sum / denominator_sum

# Step 4: Calculate the optimal intercept (b)
# Formula: b = y_mean - m * x_mean
b_best = y_mean - m_best * x_mean

# Step 5: Calculate the MSE for the best fit line
mse_best = calculate_mse(X, Y, m_best, b_best)

print("--- Best Fit Line (Optimally Tuned via Least Squares) ---")
print(f"Optimal Slope (m): {m_best:.3f}")
print(f"Optimal Intercept (b): {b_best:.3f}")
print(f"Resulting Mean Squared Error (MSE): {mse_best:.3f}")
print("-" * 50)

# Step 6: Prediction Demonstration
print("\nPrediction Demonstration:")
# Predict y for a new x value (e.g., x=12)
x_new = 12
y_predicted = m_best * x_new + b_best

print(f"The best fit line equation is: y = {m_best:.3f}x + {b_best:.3f}")
print(f"For x = {x_new}, the predicted y is: {y_predicted:.3f}")

print("\n--- Summary of Tuning Logic ---")
print("The 'tuning' goal is to minimize the MSE. We started with a guess (MSE:",
      f"{mse_guess:.3f}) and used the Least Squares formulas to mathematically",
      f"find the optimal (m, b) pair, resulting in a much lower MSE ({mse_best:.3f}).")

# Optionally, you could plot this data using a plotting library like matplotlib
# to visually demonstrate the fit, but the core logic is contained above.