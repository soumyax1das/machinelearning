Association Algorithms work on 3 fundamental attributes.

1. Support
2. Confidence
3. Lift

**Support** and **Confidence** are two simple but important metrics used in data analysis (especially in a technique called "Association Rule Mining," like what items are bought together in a store) to measure how strong a pattern or rule is.

Here's an explanation with a simple analogy:

Imagine a store that keeps track of every purchase (a "transaction"). We're looking for a rule, like "People who buy **Bread** also buy **Butter**." 🧈🍞

***

### Support: How Often Does the Combination Happen?

**Support** tells you how **popular** or **frequent** a combination of items is, relative to all the transactions.

* **Simple Idea:** It's the percentage of all shopping trips that included **both** Bread and Butter.
* **Purpose:** It shows if a combination happens often enough to matter. If the support is very low (e.g., Bread and Butter are only bought together in 1% of all transactions), the pattern is rare and not very interesting, even if the relationship is strong.

| Metric | Simple Explanation | What it Answers |
| :--- | :--- | :--- |
| **Support** | **Out of ALL shopping trips, what percentage included both A and B?** | **"How common is this entire pattern?"** |

***

### Confidence: How Likely is the Second Item, Given the First?

**Confidence** tells you, out of all the times the first item was bought, how often the second item was also bought. It measures the **reliability** of the "if-then" rule.

* **Simple Idea:** Out of all the people who bought **Bread**, what percentage of them **also** bought Butter?
* **Purpose:** It measures the predictive power of the rule. A high confidence (e.g., 90%) means that if someone buys Bread, they are very likely to also buy Butter.

| Metric | Simple Explanation | What it Answers |
| :--- | :--- | :--- |
| **Confidence** | **Out of the shopping trips that included A, what percentage also included B?** | **"If A happens, how likely is it that B will follow?"** |

***

## How They Work Together

You need both high **Support** and high **Confidence** for a rule to be truly useful:

1.  **High Support** ensures the pattern is common enough to have a business impact (it's not a rare fluke).
2.  **High Confidence** ensures the rule is reliable for making predictions (the link between the items is strong).

For the rule "If Bread, then Butter," you'd want to know:
* "Are Bread and Butter bought together often enough to justify putting them next to each other?" (**Support**)
* "When people buy Bread, how sure are we that they'll grab Butter too?" (**Confidence**)


### Lift
Lift in association rule mining is a measure that tells you how much more likely two items are to be bought together than you would expect if the decision to buy each item was completely independent. It essentially helps distinguish true, interesting relationships from events that just happen to occur often because one or both items are popular.

Here's an easy, relatable breakdown:

The Analogy: Movie Snacks

Imagine you run a movie theater concession stand, and you're analyzing customer purchases to see what items are associated with each other.

Baseline Expectation (Lift = 1):
Suppose 50\% of all customers buy Popcorn.
Suppose 10\% of all customers buy a Large Soda.
If the purchases were completely independent, you'd expect only 50\% \times 10\% = 5\% of all customers to buy both Popcorn and a Large Soda (just by random chance).
If your analysis finds that exactly 5\% of customers buy both, the Lift is 1. It means buying Popcorn has no lifting effect on the probability of buying a Large Soda.
Positive Association (Lift > 1):
Now, suppose you find that 10\% of all customers actually buy both Popcorn and a Large Soda (instead of the expected 5\%).
The Lift would be \frac{\text{Observed Co-occurrence (10\%) }}{{\text{Expected Co-occurrence (5\%)} }} = 2.
A lift of 2 means that a customer who buys Popcorn is twice as likely to also buy a Large Soda compared to a random customer from the entire population. This rule is truly interesting because the two items positively lift each other's purchase probability.
Negative Association (Lift < 1):
Now consider a different item: Diet Bar.
Suppose only 2\% of all customers buy both Popcorn and a Diet Bar. (Expected: 5\%).
The Lift would be \frac{\text{Observed Co-occurrence (2\%)}}{{\text{Expected Co-occurrence (5\%)} }} = 0.4.
A lift less than 1 suggests a negative association. It means buying Popcorn makes the purchase of a Diet Bar less likely than it would be by random chance—perhaps customers who want popcorn (an indulgence) are less inclined to buy a diet snack.
In summary:

Lift Value	Meaning	Interpretation
> 1	Positive Association	The items appear together more often than expected by chance. (This is the interesting, actionable rule!)
= 1	Independence	The items appear together exactly as often as expected by chance.
< 1	Negative Association	The items appear together less often than expected by chance.
This video provides a similar explanation of Lift in the context of association rules in machine learning. Lift- Association Rule | Machine Learning
