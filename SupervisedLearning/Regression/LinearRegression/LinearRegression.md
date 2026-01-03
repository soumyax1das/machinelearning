## 📈 What is Linear Regression?

**Linear regression** is a fundamental statistical method used to model the relationship between a **dependent variable** (the outcome you want to predict) and one or more **independent variables** (the predictors).

At its core, linear regression assumes that the relationship between these variables can be best summarized by a **straight line** (or a flat surface in higher dimensions) that *best fits* the observed data. The goal is to estimate the coefficients of the linear equation that minimize the difference between the predicted values and the actual observed values.

The simplest form, **simple linear regression**, is described by the equation:

$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:
* $Y$ is the dependent variable.
* $X$ is the independent variable.
* $\beta_0$ is the $y$-intercept (the value of $Y$ when $X=0$).
* $\beta_1$ is the slope (the change in $Y$ for every one-unit change in $X$).
* $\epsilon$ is the error term, representing the difference between the actual and predicted values.

---

## 🧐 Why is it called "Linear"?

The term **"linear"** is used because the model assumes a straight-line relationship between the variables, and crucially, because the model is **linear in its parameters** ($\beta_0$ and $\beta_1$).

* **Straight Line:** When plotted on a graph, the relationship between the independent variable and the dependent variable is modeled as a straight line.
* **Linear in Parameters:** Even in **polynomial regression** (e.g., $Y = \beta_0 + \beta_1 X + \beta_2 X^2$), where the independent variable is squared, the model is still called "linear regression" because the equation is a linear function of the unknown parameters ($\beta$ values) that are being estimated. The exponents are on the variables, not the coefficients.

---

## ↩️ Why is it called "Regression"?

The term **"regression"** has a fascinating historical origin that dates back to the 19th-century work of Sir Francis Galton, a statistician and polymath.

* **Origin: "Regression Toward the Mean":** Galton was studying heredity, specifically the relationship between the heights of parents and their children. He observed that while tall parents tended to have tall children, and short parents tended to have short children, the children's heights generally "regressed" or moved back toward the **average height of the overall population**.
* **Evolution of the Term:** Galton called this biological phenomenon **"regression toward mediocrity"** (later simplified to **"regression toward the mean"**).
* **Modern Usage:** Although the technique is now used for much more than just predicting biological averages, the name stuck. Today, **regression analysis** refers to any statistical technique used to estimate the relationship between variables and make predictions, with linear regression being the most common form.

## Watch
Stanford Lecture on Linear Regression: 
Statistical Path : [YouTube Link](https://www.youtube.com/watch?v=ZkjP5RJLQF4)
Machine Learning Course : [YouTube Link](https://www.youtube.com/watch?v=4b4MUYve_U8)

## Theory
Given a data set where we have $n$ observations of a dependent variable $Y$ and one or more independent variables $X_1, X_2, ..., X_p$, the goal of linear regression is to find the best-fitting linear equation that describes the relationship between these variables.
The general form of a multiple linear regression model is: 
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon$$
Where:
* $Y$ is the dependent variable.
* $X_1, X_2, ..., X_p$ are the independent variables.
* $\beta_0$ is the intercept.
* $\beta_1, \beta_2, ..., \beta_p$ are the coefficients for each independent variable.
* $\epsilon$ is the error term.


### Hypothesis
In linear regression, the hypothesis function is a linear combination of the input features. For a single feature, the hypothesis can be expressed as:
$$h_\theta(x) = \theta_0 + \theta_1 x$$
Where:
* $h_\theta(x)$ is the predicted value of the dependent variable.
* $\theta_0$ is the intercept (bias term).
* $\theta_1$ is the coefficient for the independent variable $x$.
* $x$ is the independent variable.
* $\theta$ represents the parameters of the model.
* For multiple features, the hypothesis can be generalized as:
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$
Where:
* $x_1, x_2, ..., x_n$ are the independent variables.
* $\theta_1, \theta_2, ..., \theta_n$ are the coefficients for each independent variable.
* $n$ is the number of features.
* The goal of linear regression is to find the optimal values of the parameters $\theta$ that minimize the difference between the predicted values $h_\theta(x)$ and the actual values of the dependent variable $Y$.
* This is typically done using methods like **Ordinary Least Squares (OLS)**, which minimizes the sum of the squared differences between the predicted and actual values.
  * The hypothesis function represents the linear relationship between the independent variables and the dependent variable, allowing us to make predictions based on the input features.
  * The parameters $\theta$ are learned from the training data during the model fitting process.
  * Once the model is trained, we can use the hypothesis function to make predictions for new data points by plugging in the values of the independent variables.
  * The hypothesis function is a key component of linear regression, as it defines the mathematical relationship that the model uses to make predictions.
  * By adjusting the parameters $\theta$, we can fit the model to the training data and capture the underlying patterns in the data.
  * Overall, the hypothesis function in linear regression provides a way to model the relationship between the independent variables and the dependent variable, allowing us to make predictions and understand the impact of each feature on the outcome.
  * The hypothesis function is a linear equation that represents the relationship between the independent variables and the dependent variable in linear regression.

### Cost Function
In linear regression, the cost function is a measure of how well the model's predictions match the actual values of the dependent variable. The most commonly used cost function for linear regression is the **Mean Squared Error (MSE)**, which calculates the average of the squared differences between the predicted values and the actual values. The cost function can be expressed mathematically as:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
Where:
* $J(\theta)$ is the cost function.
* $m$ is the number of training examples.
* $h_\theta(x^{(i)})$ is the predicted value for the $i$-th training example.
* $y^{(i)}$ is the actual value for the $i$-th training example.
* The goal of linear regression is to minimize the cost function by finding the optimal values of the parameters $\theta$. This is typically done using optimization algorithms such as **Gradient Descent** or **Normal Equation**.
* By minimizing the cost function, we can find the best-fitting line that represents the relationship between the independent variables and the dependent variable.
* The cost function provides a quantitative measure of the model's performance, allowing us to evaluate how well the model is able to predict the outcome based on the input features.
* Overall, the cost function in linear regression is a crucial component that helps us optimize the model and improve its predictive accuracy.
* The cost function is a mathematical representation of the error between the predicted values and the actual values in linear regression.



### Optimization Algorithm
To find out the optimal parameters of the coefficents in linear regression, there are mainly three methods:
1. **Normal Equation**: This method provides a closed-form solution to find the optimal parameters without the need for iterative optimization. The optimal parameters can be calculated using the formula:
   $$\theta = (X^T X)^{-1} X^T y$$
   Where:
   * $X$ is the matrix of input features.
   * $y$ is the vector of actual values.
   * $\theta$ is the vector of optimal parameters.
   * This method is efficient for small to medium-sized datasets but can be computationally expensive for large datasets due to the matrix inversion operation.
   2. **Gradient Descent**: This is an iterative optimization algorithm that updates the parameters in the direction of the steepest descent of the cost function. The update rule for gradient descent is given by:
      $$\theta := \theta - \alpha \nabla J(\theta)$$
      Where:
      * $\alpha$ is the learning rate, which controls the step size of each update.
      * $\nabla J(\theta)$ is the gradient of the cost function with respect to the parameters.
      * Gradient descent is suitable for large datasets and can converge to the optimal parameters over multiple iterations.

       **Calculus Rules:** Additional Explanation about Power Rule, Product Rule and Chain Rule that is used heavily in deriving the gradient descent algorithm:
       Power Rule: 
             If $f(x) = x^n$, then $f'(x) = n*x^{(n-1)}$

       Product Rule:
             If $f(x) = u(x) * v(x)$, then $f'(x) = u'(x) * v(x) + u(x) * v'(x)$

       Chain Rule:
             If $f(x) = g(h(x))$, then $f'(x) = g'(h(x)) * h'(x)$
       Additional vedio explanation:
             [YouTube Link](https://www.youtube.com/watch?v=jc2IthslyzM)]

       The way it works is - we try to change the value of $\theta$ by small amount and see if the change reduces the cost function $J(\theta)$. If it does, we keep changing $\theta$ in that direction until we reach a point where further changes do not reduce the cost function significantly. This point is considered as the optimal value of $\theta$.
       The update rule can be mathematically represented as the current value of $\theta$ minus the product of the learning rate $\alpha$ and the gradient(or derivative) of the cost function $\nabla J(\theta)$:
       
       $$\theta := \theta - \alpha \nabla J(\theta)$$

       Typically learning rate is set to 0.01 or 0.001. If the learning rate is too high, we may overshoot the optimal value and if it is too low, the convergence may be very slow.

       Simpler version -
       Consider there is only one reading in the sample    
       Then the cost function is 

       $$\nabla J(\theta) = \nabla  (\frac{1}{2} *  (h_\theta(x) - y)^2)$$
       consider $u(x) = (h_\theta(x) - y)$
       and $v = u^2$
             If $f(x) = v(u(x))$, then $f'(x) = v'(u(x)) * u'(x)$
       then the cost function becomes
   $$\nabla J(\theta) = \nabla  (\frac{1}{2} *  v(u(x)))$$
     
       Using Chain Rule, we get
       = $$ \frac{1}{2} * \nabla v * \nabla u$$
       Now, using Power Rule for calculating derivative of v=u^2, we get
       = $$ \frac{1}{2} * (2 * u) * \nabla u $$
       substituting back the value of u, we get
       = $$ \frac{1}{2} * 2 * (h_\theta(x) - y) * \nabla (h_\theta(x) - y) $$
       Now, using Power Rule, we get
       = $$ \frac{1}{2} * 2 * (h_\theta(x) - y) * 1 * \nabla h_\theta(x) $$
       substituting the value of $h_\theta(x) = \theta^T * x$, we get
       = $$ \frac{1}{2} * 2 * (h_\theta(x) - y) * 1 * x $$
       Simplifying, we get
       = $$ (h_\theta(x) - y) * x $$

       Using Power Rule and Chain Rule, we get
       = $$ 2 * \frac{1}{2} * \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) * x^{(i)}$$


3. **Stochastic Gradient Descent (SGD)**: This is a variant of gradient descent that updates the parameters using a single training example at a time. The update rule for SGD is given by:
   $$\theta := \theta - \alpha \nabla J(\theta; x^{(i)}, y^{(i)})$$
   Where:
   * $x^{(i)}$ and $y^{(i)}$ are the input features and actual value for the $i$-th training example.
     * SGD is particularly useful for very large datasets and can converge faster than batch gradient descent, although it may introduce more noise in the updates.
Each of these optimization algorithms has its own advantages and disadvantages, and the choice of method depends on the specific characteristics of the dataset and the computational resources available.
     * Overall, these optimization algorithms play a crucial role in finding the optimal parameters for linear regression models, allowing us to make accurate predictions based on the input features.
     * By selecting the appropriate optimization algorithm, we can effectively train linear regression models and improve their performance on various tasks.
     * The optimization algorithm is a key component of linear regression that helps us find the best-fitting line for the data by minimizing the cost function.
     * Different optimization algorithms can be used depending on the size of the dataset and the computational resources available.
     * The choice of optimization algorithm can impact the convergence speed and accuracy of the model.
     

## Comparison of methods to find optimal parameters(optimal loss function value)
The comparison of  four concepts 
- LSM (Least Squares Method)
- Normal Equation
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)

for finding linear regression coefficients falls into two main categories: **Closed-Form Solutions** and **Iterative Optimization Algorithms**.

The **Least Squares Method** defines the goal, and the **Normal Equation** is the direct solution. **Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)** are two variants of the iterative method used to reach that goal.

---

## 1. Closed-Form Solutions (Least Squares / Normal Equation)

The **Least Squares Method (LSM)** is the principle that dictates the coefficients must be chosen to minimize the **Sum of Squared Errors (SSE)** between the predicted and actual values.

The **Normal Equation** is the **analytical formula** derived from setting the partial derivatives of the SSE cost function to zero, giving the *exact* solution for the coefficients ($\theta$) in a single, non-iterative step.

| Feature | Least Squares Method (LSM) | Normal Equation |
| :--- | :--- | :--- |
| **Concept** | **Objective/Principle:** The goal is to minimize the sum of squared differences (errors) between observed and predicted values. | **Method/Formula:** The direct, closed-form matrix solution that achieves the LSM objective. |
| **Formula** | $J(\theta) = \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$ (The function to minimize) | $\theta = (X^T X)^{-1} X^T y$ (The solution) |
| **Iterative?** | **No.** Computes the result directly. | **No.** Computes the result directly. |
| **Computational Cost** | High. Dominated by the matrix inversion, $O(m^3)$, where $m$ is the number of features. | High. Dominated by the matrix inversion, $O(m^3)$, where $m$ is the number of features. |
| **Feature Scaling** | **Not Required.** | **Not Required.** |
| **Best Used When** | The number of features ($m$) is **small** (e.g., $m < 10,000$). |

---

## 2. Iterative Optimization Algorithms (GD / SGD)

Both Gradient Descent (GD) and Stochastic Gradient Descent (SGD) are algorithms that iteratively search for the coefficient values that minimize the Least Squares (SSE) objective function. They differ in how much data is used in each update step.

| Feature | Gradient Descent (GD) (aka Batch GD) | Stochastic Gradient Descent (SGD) |
| :--- | :--- | :--- |
| **Update Mechanism** | **Batch:** Computes the gradient using the **entire training dataset** for every single step. | **Stochastic:** Computes the gradient using **only one training example** (or a very small, random subset) for every single step. |
| **Update Frequency** | **Slow:** One update per **Epoch** (one full pass over the entire training data). | **Fast:** One update per **example** (many updates per epoch). |
| **Convergence Path** | **Smooth/Direct:** Takes a stable path straight to the minimum, resulting in a very precise solution. | **Noisy/Oscillating:** Takes a zig-zag path due to the variance of single-example gradients, but still converges near the minimum. |
| **Speed** | **Slower** per update, especially with a large number of data points ($n$). | **Much Faster** per update, making it better for very large datasets ($n$). |
| **Feature Scaling** | **Required.** Needed for faster and reliable convergence. | **Required.** Needed for faster and reliable convergence. |
| **Best Used When** | The number of data points ($n$) is **small to medium**, and maximum precision is required. | The number of data points ($n$) is **very large** (cannot fit in memory), and speed is paramount. |

---

## ⚖️ Summary of Computational Efficiency

The choice depends on the size of your dataset ($n$) and the number of features ($m$):

| Method | Computational Complexity | Use Case |
| :--- | :--- | :--- |
| **Normal Equation** | $O(m^3)$ | **Small $m$ (features)**. Non-iterative, exact solution. |
| **Batch GD** | $O(n \cdot m \cdot \text{iterations})$ | **Small $n$ (data)**. Accurate and stable convergence. |
| **Stochastic GD** | $O(m \cdot \text{iterations})$ (Faster because $n$ is effectively 1 per update) | **Large $n$ (data)**. Faster updates, but noisy convergence path. |

**Mini-Batch Gradient Descent (MBGD)**, which uses a small, fixed-size subset (a batch) of the data for each update, is often the **practical industry standard**, offering a balance between the stable convergence of GD and the speed of SGD.