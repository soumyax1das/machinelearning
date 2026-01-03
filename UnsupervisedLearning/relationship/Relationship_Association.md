##Mining multi level association rules and most common algorithms
The most common and fundamental **association rule algorithms** in machine learning are:

1.  **Apriori Algorithm**
2.  **FP-Growth (Frequent Pattern Growth) Algorithm**
3.  **Eclat (Equivalence Class Transformation) Algorithm**

These algorithms are primarily used in **Association Rule Mining**, an unsupervised technique to find frequent patterns, correlations, and relationships among a set of items in a database, often associated with **Market Basket Analysis** (e.g., discovering that customers who buy diapers also tend to buy beer).

***

## 1. Apriori Algorithm
The **Apriori** algorithm is the classic and most widely known algorithm.

* **Approach:** It uses a **bottom-up, breadth-first search** approach and the **Apriori property** (or anti-monotonic property): if an itemset is frequent, then all of its subsets must also be frequent. Conversely, if an itemset is infrequent, all of its supersets must also be infrequent.
* **Process:** It iteratively generates candidate itemsets of length $k$ from frequent itemsets of length $k-1$ and then prunes the infrequent candidates by scanning the entire dataset in each pass.
* **Best Use Case:** It's often used for **small datasets** and for learning the core concepts of association rule mining due to its simplicity.
* **Limitation:** It is computationally expensive and slow for very large datasets because it requires **multiple passes** over the database and generates a large number of candidate itemsets.

***

## 2. FP-Growth Algorithm
The **FP-Growth** algorithm was developed to overcome the performance limitations of Apriori.

* **Approach:** It uses a **divide-and-conquer** approach by storing the dataset's frequent patterns in a compressed tree structure called an **FP-Tree** (Frequent Pattern Tree).
* **Process:** It scans the database **only twice** to build the FP-Tree and then recursively mines the tree to generate frequent itemsets, avoiding the need to generate candidate sets entirely.
* **Best Use Case:** It is significantly **faster** and more memory efficient than Apriori, making it the preferred choice for **large-scale pattern mining** and big datasets.

***

## 3. Eclat Algorithm
The **Eclat** algorithm is a simpler alternative to Apriori.

* **Approach:** It uses a **depth-first search** strategy and a **vertical data format** (TID-list), where each item is associated with a list of transaction IDs (TIDs) in which it appears.
* **Process:** It finds the support of an itemset by simply performing an **intersection** of the TID lists of its subsets.
* **Best Use Case:** It generally performs well on **medium-sized** or **dense datasets** and can be faster than Apriori due to its use of set intersections, which are often faster than scanning the entire database.

***
This video tutorial provides a deep dive into the concept of Association Rule Learning, which is where these algorithms are used: [Association Rule Learning | Machine Learning Tutorial](https://www.youtube.com/watch?v=Bl5dGOLmF0k).
http://googleusercontent.com/youtube_content/1
