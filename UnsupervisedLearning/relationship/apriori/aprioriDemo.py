import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


dataset = [
           ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
          ]


# Get the total count of the transactions in the dataset
N = len(dataset)

# The TransactionEncoder is a preprocessing class from the $\text{mlxtend}$ Python library that converts
# a list of lists representing transaction data into a one-hot encoded NumPy array (often converted to a pandas DataFrame).
# Its primary purpose is to transform data from a common "transactional" format into the binary format required for
# Association Rule Mining algorithms, such as the Apriori algorithm.
te = TransactionEncoder()

#te.fit(dataset): This step scans all 5 transactions to identify every unique item present in the entire dataset.
# These unique items become the "features" (columns) in the final array.Unique Items Identified:
# $\text{Milk}$, $\text{Onion}$, $\text{Nutmeg}$, $\text{Kidney Beans}$, $\text{Eggs}$, $\text{Yogurt}$, $\text{Dill}$, $\text{Apple}$, $\text{Unicorn}$, $\text{Corn}$, $\text{Ice cream}$.
# te.transform(dataset): This step then converts the original list of lists into the final binary array (te_ary).
te_ary = te.fit(dataset).transform(dataset)
print(te_ary)
print(te.columns_)

print(type(te_ary))


df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)


# Pre-requisite
# ======================================================
# Define
# minimum Support Value and (MINSUP)
# minimum Confidence value (MINCONF)
MIN_SUPPORT = 0.6
MIN_CONFIDENCE = 0.5

# --- Step 2: Find Frequent Itemsets using Apriori ---
# Create frequent itemset using apriori method, providing min support value
freq_itemset = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
# Enhance freq_itemset to display item in the itemset and total count of the itemset
freq_itemset["support_count"] = freq_itemset["support"] * N
freq_itemset["item_count"] = freq_itemset["itemsets"].apply(lambda x: len(x))
print(freq_itemset)

# --- Step 3: Generate Association Rules ---
# The association_rules function takes the frequent itemsets and the metric for evaluation.
rules = association_rules(freq_itemset,metric="confidence",min_threshold=MIN_CONFIDENCE)
print(rules)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False))
