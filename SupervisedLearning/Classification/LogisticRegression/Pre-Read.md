# Pre-Read: Probability, Odds, and Log-Odds in Logistic Regression

This document explains the foundational concepts of probability and odds, which are central to understanding how Logistic Regression works.

Watch -
https://www.youtube.com/watch?v=ARfXDSkQf1Y


---

## 1. What is Probability?

**Probability** is a measure of the likelihood that an event will occur. It quantifies certainty and is expressed as a number between 0 and 1.

- **Range:** `[0, 1]`
  - **0** means the event is impossible.
  - **1** means the event is certain.
  - **0.5** means the event has an equal chance of occurring or not occurring.

- **Formula:**
  ```
  P(event) = (Number of Favorable Outcomes) / (Total Number of Possible Outcomes)
  ```

- **Example:**
  In a standard deck of 52 cards, the probability of drawing a King is:
  `P(King) = 4 / 52 ≈ 0.077` or `7.7%`

---

## 2. What are Odds?

**Odds** represent the ratio of the probability of an event occurring to the probability of it not occurring. It's another way to express the likelihood of an event.

- **Range:** `[0, ∞)` (from 0 to infinity)
  - Odds of **0** mean the event never occurs.
  - Odds of **1** mean the event has an equal chance of occurring or not (equivalent to a probability of 0.5).
  - Odds greater than **1** mean the event is more likely to occur than not.
  - Odds less than **1** mean the event is less likely to occur than not.

- **Formula:**
  ```
  Odds(event) = P(event) / (1 - P(event))
  ```
  Where `(1 - P(event))` is the probability of the event *not* occurring.

- **Example:**
  The probability of drawing a King is `4/52`. The probability of *not* drawing a King is `48/52`.
  `Odds(King) = (4/52) / (48/52) = 4 / 48 = 1 / 12`
  This is often stated as "1 to 12 odds in favor of drawing a King."

---

## 3. Why is Probability Not Enough for Linear Models?

Linear regression models a direct linear relationship: `Y = β₀ + β₁X`. If we try to use this to predict a probability (`P`), we run into two major problems:

1.  **The Bounded Range Problem:**
    - Probability is strictly bounded between **0 and 1**.
    - The output of a linear equation (`β₀ + β₁X`) is **unbounded**—it can range from `-∞` to `+∞`.
    - A linear model could easily predict a "probability" of -0.2 or 1.5, which is nonsensical.

2.  **The Non-Linear Relationship Problem:**
    - The effect of a feature on probability is often not linear. For instance, an extra hour of studying might increase your probability of passing an exam from 5% to 15% (a 10% jump). However, if your probability is already 90%, that same extra hour might only increase it to 95% (a 5% jump).
    - A linear model assumes a constant effect. It assumes that for every one-unit increase in `X`, `Y` changes by a constant amount (`β₁`). This assumption doesn't hold true for probabilities.

Because of these issues, we cannot directly model probability with a linear equation.

---

## 4. The Use of Odds and Log-Odds

To solve the problems above, Logistic Regression introduces two transformations: **Odds** and then **Log-Odds**.

### Step 1: From Probability to Odds

- We first convert the probability to **Odds**.
- As we saw, `Odds = P / (1 - P)`.
- The range of Odds is `[0, ∞)`.
- This solves half the problem: we no longer have an upper bound of 1. However, we still have a lower bound of 0, so the range is not yet suitable for a linear model.

### Step 2: From Odds to Log-Odds (The "Logit")

- Next, we take the natural logarithm of the odds. This is called the **Logit function**.
- **`Log-Odds = log(Odds) = log(P / (1 - P))`**
- The range of Log-Odds is `(-∞, +∞)`.

**This is the crucial step!** The Log-Odds scale is unbounded on both ends, just like the output of a linear equation.

### The Core of Logistic Regression

Logistic Regression models the **Log-Odds** of an event as a linear combination of the features.

`log(P / (1 - P)) = β₀ + β₁X₁ + ... + βₙXₙ`

This is why it's called a "generalized linear model." It uses a linear equation to predict a *transformation* of the probability, not the probability itself.

To get the final probability, the model reverses the transformation using the **Sigmoid (or Logistic) function**:

`P = 1 / (1 + e^-(β₀ + β₁X₁ + ...))`

This function takes any real-valued number and maps it back to the `[0, 1]` range, giving us a valid probability.

