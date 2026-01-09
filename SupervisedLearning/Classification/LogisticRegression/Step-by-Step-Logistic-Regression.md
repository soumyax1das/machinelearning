# Step-by-Step Guide to Logistic Regression

This document walks through the process of building a logistic regression model from the ground up, using a practical example.

**Example code** https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial

**Scenario:** We want to predict whether a student will pass an exam (`passed`). Our prediction will be based on three numerical features:
- `hrs_of_study` (hours spent studying)
- `hrs_of_sleep` (hours of sleep per night)
- `days_absent` (number of days absent from class)

The `passed` column is binary: `1` for passed, `0` for failed.

---

### Step 1: Start with a Linear Equation (The Wrong Approach)

If we were doing linear regression, we would model the outcome directly:

`passed = β₀ + β₁ * hrs_of_study + β₂ * hrs_of_sleep + β₃ * days_absent`

- `β₀` is the intercept (the baseline outcome).
- `β₁`, `β₂`, `β₃` are the coefficients that represent the weight of each feature.

**Problem:** The output of this equation is a continuous, unbounded number (from -∞ to +∞). However, our desired output is a **probability** of passing, which must be between 0 and 1. A linear model might predict a "probability" of 1.7 or -0.4, which is meaningless.

---

### Step 2: The Goal - We Need a Probability

Our goal is to predict `P(passed=1)`, the probability that a student passes. We need a function that takes the linear equation's output and maps it to the `[0, 1]` range.

To get there, we introduce the concepts of **Odds** and **Log-Odds**.

---

### Step 3: Convert Probability to Odds

**Odds** are the ratio of the probability of an event happening to the probability of it not happening.

- **Formula:** `Odds = P(passed=1) / P(passed=0)`
- Since `P(passed=0) = 1 - P(passed=1)`, we can write:
  `Odds = P / (1 - P)`

- **Range:**
  - If `P = 0.5`, `Odds = 0.5 / 0.5 = 1`. (Equal odds)
  - If `P > 0.5`, `Odds > 1`. (More likely to pass)
  - If `P < 0.5`, `Odds < 1`. (Less likely to pass)
  - The range of Odds is `[0, ∞)`.

This is a step in the right direction. We've removed the upper bound of 1, but we still have a lower bound of 0. The output is not yet symmetrical around zero.

---

### Step 4: Convert Odds to Log-Odds (The "Logit" Function)

To get an unbounded and symmetrical range, we take the natural logarithm of the odds. This is the **Logit** transformation.

- **Formula:** `Log-Odds = log(Odds) = log(P / (1 - P))`

- **Range:**
  - If `P = 0.5`, `Odds = 1`, and `log(1) = 0`.
  - If `P -> 1`, `Odds -> ∞`, and `log(Odds) -> ∞`.
  - If `P -> 0`, `Odds -> 0`, and `log(Odds) -> -∞`.
  - The range of Log-Odds is `(-∞, ∞)`.

**This is the key insight!** The Log-Odds scale is unbounded, just like the output of our linear equation. We can now connect the two.

---

### Step 5: Build the Logistic Regression Model

The core of logistic regression is to model the **Log-Odds** as a linear combination of the features.

`log(P / (1 - P)) = β₀ + β₁ * hrs_of_study + β₂ * hrs_of_sleep + β₃ * days_absent`

The model finds the best coefficient values (`β₀`, `β₁`, `β₂`, `β₃`) that make this equation true, typically using a method called Maximum Likelihood Estimation.

Let's assume the training process gives us the following coefficients:
- `β₀ = -4.0` (Intercept)
- `β₁ = 0.8` (for `hrs_of_study`)
- `β₂ = 0.2` (for `hrs_of_sleep`)
- `β₃ = -0.5` (for `days_absent`)

Our final model is:
`Log-Odds = -4.0 + 0.8 * hrs_of_study + 0.2 * hrs_of_sleep - 0.5 * days_absent`

---

### Step 6: Make a Prediction (From Log-Odds back to Probability)

Now, let's predict the outcome for a new student:
- `hrs_of_study = 6`
- `hrs_of_sleep = 7`
- `days_absent = 1`

1.  **Calculate the Log-Odds:**
    `Log-Odds = -4.0 + 0.8 * (6) + 0.2 * (7) - 0.5 * (1)`
    `Log-Odds = -4.0 + 4.8 + 1.4 - 0.5`
    `Log-Odds = 1.7`

This value, `1.7`, is the predicted log-odds of the student passing.

2.  **Convert Log-Odds to Probability using the Sigmoid Function:**
    To get the probability, we must reverse the logit transformation. The inverse of the logit function is the **Sigmoid** (or **Logistic**) function.

    - **Formula:** `P = 1 / (1 + e^(-Log-Odds))`

    - **Calculation:**
      `P(passed=1) = 1 / (1 + e^(-1.7))`
      `P(passed=1) = 1 / (1 + 0.1827)`
      `P(passed=1) = 1 / 1.1827`
      `P(passed=1) ≈ 0.845`

**Prediction:** The model predicts an **84.5% probability** that this student will pass. We can set a threshold (e.g., 0.5) to make a final decision: since 0.845 > 0.5, we classify the student as `passed=1`.

---

### Step 7: Interpreting the Model (The Odds Ratio)

The coefficients (`β`) are in terms of log-odds, which are not very intuitive. To interpret them, we use the **Odds Ratio** by taking the exponent of the coefficient: `e^β`.

- **`hrs_of_study` (`β₁ = 0.8`):**
  - `Odds Ratio = e^0.8 ≈ 2.22`
  - **Interpretation:** For each additional hour a student studies, the **odds** of them passing are multiplied by **2.22** (i.e., they increase by 122%), holding sleep and absences constant.

- **`hrs_of_sleep` (`β₂ = 0.2`):**
  - `Odds Ratio = e^0.2 ≈ 1.22`
  - **Interpretation:** For each additional hour of sleep, the odds of passing increase by a factor of **1.22** (a 22% increase), holding other variables constant.

- **`days_absent` (`β₃ = -0.5`):**
  - `Odds Ratio = e^-0.5 ≈ 0.61`
  - **Interpretation:** For each additional day a student is absent, the odds of passing are multiplied by **0.61** (i.e., they decrease by 39%), holding other variables constant.

This ratio of odds is a powerful way to understand the influence of each feature on the outcome.

---

### Summary Flowchart

1.  **Problem:** Predict a binary outcome (0 or 1).
2.  **Challenge:** Linear models produce unbounded output `(-∞, ∞)`. Probability is bounded `[0, 1]`.
3.  **Transformation 1 (Odds):** Convert probability `P` to `Odds = P / (1 - P)`. Range becomes `[0, ∞)`.
4.  **Transformation 2 (Log-Odds):** Take the log of odds: `log(Odds)`. Range becomes `(-∞, ∞)`.
5.  **Modeling:** Equate the linear model to the log-odds: `log(P / (1-P)) = β₀ + β₁X₁ + ...`
6.  **Prediction:** Calculate the log-odds for new data, then use the **Sigmoid function** `P = 1 / (1 + e^(-log_odds))` to get the final probability.
7.  **Interpretation:** Use the **Odds Ratio** (`e^β`) to explain how a one-unit change in a feature affects the odds of the outcome.

