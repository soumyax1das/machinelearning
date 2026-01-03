Prerequisite Knowledge
============================
##Basic theory of Decision Tree
Reading:
https://www.cs.toronto.edu/~axgao/cs486686_f21/lecture_notes/Lecture_07_on_Decision_Trees.pdf

Recording:

https://www.youtube.com/watch?v=_L39rN6gz7Y

Decision trees are models which essentially represents a logical tree where each level
of the tree is formed based on the decision taken on the value of a specific feature column.
Based on the unique values in the feature column child nodes are created from parent node.
A **Decision Tree (DT)** in machine learning is a **non-parametric supervised learning algorithm** used for both **classification** and **regression** tasks. It works by learning simple **if-then-else decision rules** inferred from the data features, creating a model that resembles a flowchart or an upside-down tree structure.

---

## 🌳 Decision Tree Structure

The structure of a decision tree consists of several key components:

* **Root Node:** Represents the entire dataset and is the starting point. It's the best feature to split the data on.
* **Internal Nodes (Decision Nodes):** Represent a test on an attribute (a feature) in the dataset. Each branch leading from an internal node represents the outcome of that test.
* **Branches:** Represent the decision rules or the possible outcomes of the test at the node.
* **Leaf Nodes (Terminal Nodes):** Represent the final outcome, which is a class label (for classification) or a numerical value (for regression).

The goal of the algorithm is to recursively partition the data into subsets that are as **"pure"** as possible (meaning the nodes contain data points mostly belonging to one class). It selects the best feature to split on at each step using metrics like **Information Gain** (based on **Entropy**) or **Gini Impurity**.

---

## 📝 Example: Predicting if a Person will Play Tennis

Imagine you have a dataset with features like **Weather**, **Temperature**, **Humidity**, and **Wind** to predict whether a person will **Play Tennis** or not (Yes/No).

### Initial Data and The Root

The algorithm starts at the **Root Node** with the entire dataset. It then determines which feature offers the best split (the largest reduction in impurity) to make the resulting subsets purer. Let's assume the algorithm finds that **Weather** is the most informative feature for the first split.

### First Split (Root Node)

The tree splits the data based on the values of the **Weather** feature:

* **Root Node: Weather**
    * **Branch 1: Sunny** $\rightarrow$ Leads to a new Decision Node (Internal Node).
    * **Branch 2: Overcast** $\rightarrow$ Leads directly to a **Leaf Node** (e.g., "Play Tennis: Yes" in a perfectly pure split).
    * **Branch 3: Rainy** $\rightarrow$ Leads to a new Decision Node (Internal Node).

### Subsequent Splits (Internal Nodes)

For the branches that are not yet "pure" (e.g., "Sunny" and "Rainy"), the algorithm repeats the process, choosing the next best feature to split that subset of data.

* **Internal Node (from "Sunny" branch): Humidity**
    * **Branch: High** $\rightarrow$ **Leaf Node:** "Play Tennis: No"
    * **Branch: Normal** $\rightarrow$ **Leaf Node:** "Play Tennis: Yes"

* **Internal Node (from "Rainy" branch): Wind**
    * **Branch: Strong** $\rightarrow$ **Leaf Node:** "Play Tennis: No"
    * **Branch: Weak** $\rightarrow$ **Leaf Node:** "Play Tennis: Yes"

### Final Decision Tree

The final structure is a series of questions:

> **IF** Weather is Overcast, **THEN** Play Tennis = Yes.
> **ELSE IF** Weather is Sunny **AND** Humidity is Normal, **THEN** Play Tennis = Yes.
> **ELSE IF** Weather is Sunny **AND** Humidity is High, **THEN** Play Tennis = No.
> **ELSE IF** Weather is Rainy **AND** Wind is Weak, **THEN** Play Tennis = Yes.
> **ELSE IF** Weather is Rainy **AND** Wind is Strong, **THEN** Play Tennis = No.

### Using the Tree for Prediction

To predict whether a new person will play tennis when the weather is **Sunny** and the humidity is **High**:

1.  Start at the **Root Node** (**Weather**).
2.  Follow the **Sunny** branch.
3.  Go to the **Internal Node** (**Humidity**).
4.  Follow the **High** branch.
5.  Reach the **Leaf Node** with the final prediction: **No** (Play Tennis).

Decision trees are highly **interpretable** because their logic is a clear, visual sequence of rules, making it easy to understand *why* a certain prediction was made.

## Exit Criteria of Splitting a decision Tree
A decision tree is built by splitting the nodes into multiple child nodes based on features.
However there are complexities in splitting if there are hundreds of features. So there are certain rules
to guide when to stop splitting.

There are three possible stopping criteria for the decision tree algorithm. 

#### a) When all the examples of records belong to same class
In this case splitting more won't change the outcome. As the result has only one value. Remaining features
won't add any value, if used for splitting.

#### b) When there is no feature left
If there is no feature left to split, then the algorithm stops splitting.

Noisy Data : When the last feature splits the data and under one child node we have two or more observations
of the dependent feature, then we hit a road block. We know we have multiple observations, say True or False, but we
can't split as we don't have enough feature. So the data becomes noisy here. With noisy data, even if we know the
values of all the input features, we are still unable to make a deterministic decision.
One reason for having a noisy data set is that the decision may be influenced by
some features that we do not observe.

#### c) When there is no more example left
If a parent node(which is not root node) is split based on a feature which has 3 potential values in the training
dataset, but the 3rd child may not have an example with the parent node, child node combination. In this case there
is no point in splitting as there is not enough information. In this case multiple approaches can be taken -
a) Evaluate from the remaining tree 
b) Adding more data to the training dataset


#### Algortihm for Splitting
Algorithm 1 Decision Tree Learner (examples, features)

  if all examples are in the same class then

    return the class label.

  else if no features left then

    return the majority decision.

  else if no examples left then

    return the majority decision at the parent node.

  else

      choose a feature f.

  for each value v of feature f do

       build edge with label v.

       build sub-tree using examples where the value of f is v.



## Choosing the right feature to split Decision Tree 
When we have a dataset where the dependent feature is D and it has i independent feature
C-1 to C-i, the tree is split for all or subset of {C-1,...,C-i}. The order of split is important
for efficiency purposes.

The order in which a splitting feature is chosen depneds on

  "lowest value of surprise" or "highest homegenity" or "lowest impurity"

For example if C-1 splits the data into two equal amount of claases of D, and C-2 splits the data into
one lasrge class and one very small class, the suprise factor of split based on C-2 is less and homogenity
is more, so C-2 is chosen over C-1.
There are two primary ways to choose features -

  a) Entropy and Information Gain

  b) Gini Index

Watch - https://www.youtube.com/watch?v=wefc_36d5mU

Probability and Expected Value
Surprise and Entropy - https://www.youtube.com/watch?v=YtebGVx-Fxw&t=575s
####Information Gain and Entropy
Reading -
https://medium.com/codex/decision-tree-for-classification-entropy-and-information-gain-cd9f99a26e0d

Entropy is a measure of disorder or impurity in the given dataset.


#### Gini Index, 

Watch - https://www.youtube.com/watch?v=41SHQjwuQ5o



The Gini Index, often called Gini Impurity, is a metric used in decision tree algorithms (like CART, Classification and Regression Trees) to measure the impurity or disorder within a set of data.

In the context of a decision tree, the Gini Index helps determine the best feature and split point at each node by quantifying how "mixed" the class labels are in that node.

📉 What the Gini Index Represents
The Gini Index is a value between 0 and 1 (though in binary classification, the maximum is 0.5):

Gini Index Value	

Low Value 0	- Perfect Purity. 

All samples in the node belong to the same class. This is the ideal goal for leaf nodes.

High Value (e.g., 0.5)	- Maximum Impurity. 

Samples are equally distributed among different classes (e.g., 50% class A and 50% class B in a binary problem).
The goal of the decision tree algorithm is to choose the split that results in the lowest possible Gini Index (highest reduction in impurity) for the resulting child nodes.

🔢 How the Gini Index is Calculated

The Gini Impurity for a single node (G) is calculated based on the probability of an item belonging to each class.

1. Gini Impurity Formula for a Node

For a node containing C different classes, the Gini Impurity(G) is calculated as:

G = 1− SumOf( (P<sub>i</sub>)<sup>2</sup> ), where i = 1 to C



Where:

i is the total number of unique values or classes of the dependent column/feature.

P<sub>i</sub> is the probability (or proportion) of samples belonging to class i at that specific node. 



P<sub>i</sub> =
(Number of samples in class i) / (Total number of samples in the node)



Example: Calculating Gini for a Node

Imagine a node with a total of 10 samples for a binary classification problem (Class 'Yes' or 'No'):

5 samples are Class 'Yes' (p<sub>1</sub>
=5/10=0.5)

5 samples are Class 'No' (p<sub>2</sub>
=5/10=0.5)

G=1−((0.5)<sup>2</sup> + (0.5)<sup>2</sup>)
G=1−(0.25+0.25)
G=1−0.5=0.5
A Gini Index of 0.5 is the maximum impurity for a binary split, meaning the node is perfectly mixed.

2. Gini Index (Weighted) for a Split

When evaluating a potential split on a feature, the algorithm calculates the Gini Index for the entire split by taking a weighted average of the Gini Impurity of the child nodes it creates.

For a feature that splits the parent node (P) into k child nodes (
C<sub>1</sub>,C<sub>2</sub>,…,C<sub>k</sub>):

G<sub>split</sub>
= SumOf ((n<sub>j</sub>/n)*G(C<sub>j</sub>)) for all j, where j is the unique values of the dependent feature

Where:

n is the total number of samples in the parent node P.

n<sub>j</sub> 
is the number of samples in the child node C<sub>j</sub>

.

G(C<sub>j</sub>) is the Gini Impurity of the child node C

The decision tree selects the feature and split point that yields the minimum G<sub>split</sub>
value. This split is considered the "best" because it creates the purest child nodes possible.



####Gini Impurity
https://medium.com/codex/decision-tree-for-classification-entropy-and-information-gain-cd9f99a26e0d

Reduction in variance

Occam’s razor principle

## Dealing with over-fitting with pruning

It is always better to have a smaller tree that covers most instead of having a full blown tree with overfitting.
The smaller and shallower tree may not predict all of the training data points perfectly but it may generalize to test data better.
At a high level we have two options to prevent over-fitting when learning a decision tree:

a) Pre-pruning

b) Post-pruning

### Pre-pruning Strategy
For pre-pruning, the model will split the examples at a node only when it’s useful. 
Following are well known strategies

1. Maximum depth: We can decide not to split the examples if the depth of that node
   has reached a maximum value that we decided beforehand.
   
2. Minimum number of examples at the leaf node: We can decide not to split the examples
   if the number of examples remaining at that node is less than a predefined threshold
   value.

3. Minimum information gain: We can decide not to split the examples if the benefit of
   splitting at that node is not large enough. We can measure the benefit by calculating
   the expected information gain. In other words, do not split examples if the expected
   information gain is less than the threshold.
   
4. Reduction in training error: We can decide not to split the examples at a node if the
   reduction in training error is less than a predefined threshold value.


### Post-pruning Strategy
Post-pruning is particularly useful when any individual feature is not informative, but mul-
tiple features working together is very informative.

Note that we will only decide
to post-prune a node if it has only leaf nodes as its descendants.



## Decision tree with categorical and numerical features (Mixed Features)




## Evaluating the model

Confusion Matrix

https://www.youtube.com/watch?v=Kdsp6soqA7o

A **confusion matrix** is a fundamental performance evaluation tool in **machine learning**, used to summarize the results of a **classification algorithm**. It's a table layout that provides a detailed breakdown of correct and incorrect predictions made by the model, comparing the model's predictions to the actual (ground truth) values.

---

### 🏗️ Structure of a Binary Confusion Matrix

For a binary classification problem (e.g., classifying an email as "Spam" or "Not Spam," or a patient as "Sick" or "Healthy"), the confusion matrix is a **2x2 table** composed of four key components:


| | **Actual Positive** | **Actual Negative** |
| :---: | :---: | :---: |
| **Predicted Positive** | **True Positive (TP)** | **False Positive (FP)** |
| **Predicted Negative** | **False Positive (FN)** | **False Negative (TN)** |


### **The Four Outcomes**

1.  **True Positive (TP):** The model **correctly** predicted the positive class.
  * *Example:* The model correctly predicted a spam email as **Spam**.
2.  **True Negative (TN):** The model **correctly** predicted the negative class.
  * *Example:* The model correctly predicted a regular email as **Not Spam**.
3.  **False Positive (FP):** The model **incorrectly** predicted the positive class. (A **Type I Error** or "False Alarm").
  * *Example:* The model incorrectly predicted a regular email as **Spam**.
4.  **False Negative (FN):** The model **incorrectly** predicted the negative class. (A **Type II Error** or "Missed Case").
  * *Example:* The model incorrectly predicted a spam email as **Not Spam**.

---

### 💡 Why is it Important?

Simply looking at **Accuracy** (the total number of correct predictions: $(\text{TP} + \text{TN}) / \text{Total}$) can be misleading, especially with **imbalanced datasets**.

The confusion matrix provides the raw numbers needed to calculate more nuanced metrics that highlight specific types of errors:

* **Precision:** 
  
Focuses on the accuracy of the **positive predictions** 
  
    ({TP}) / ( {TP} + {FP} ). 
It matters when **False Positives** are costly (e.g., wrongly flagging an innocent person for fraud).

* **Recall** (or Sensitivity): 
  
Focuses on the model's ability to find **all actual positive cases** 

({TP} / ({TP} + {FN})). 

It matters when **False Negatives** are costly (e.g., failing to diagnose a patient with a serious disease).

* **F1-Score:** 
  
The harmonic mean of Precision and Recall, providing a single metric that balances both.

    {F1 Score} = { {Precision} * {Recall} } / { {Precision} + {Recall} }

When to Use: It is the preferred metric for imbalanced datasets because it penalizes models that favor one metric (like achieving high Recall by accepting many FPs, or high Precision by accepting many FNs).21 A model with a high F1 Score must have a good balance of both Precision and Recall.22

The confusion matrix makes it clear where the model is "confused" (misclassifying one class as another), allowing data scientists to diagnose problems and tune the model based on the cost of different error types.

---

For a simple visual explanation of the confusion matrix, watch [Confusion Matrix Fundamentals Explained in 3 MINUTES!](https://www.youtube.com/watch?v=NhvZT_uCYDI).
http://googleusercontent.com/youtube_content/1