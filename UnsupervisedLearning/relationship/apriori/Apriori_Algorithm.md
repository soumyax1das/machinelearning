
Study Guide
======================================================================
Recording -
https://www.youtube.com/watch?v=NT6beZBYbmU

Reading -
https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
https://www3.cs.stonybrook.edu/~cse521/ch5book.pdf
https://www.datacamp.com/tutorial/apriori-algorithm




This is a demonstration of apriori algorithm to find association rules.
Apriori algorithm works in multiple phases.


Important Note
==================
The Apriori algorithm works in two main phases:

Pre-requisite
======================================================
    Define
    minimum Support Value and (MINSUP)
    minimum Confidence value (MINCONF)


Phase 1 — Find all frequent itemsets
======================================================
It starts from:

Frequent 1-itemsets →
then generates 2-itemset candidates from them → checks their support → keeps frequent ones →
break if found frequent itselt at 2-itemset is 0
then generates 3-itemsets candidates from them → checks their support → keeps frequent ones →
break if found frequent itselt at 3-itemset is 0
...
...
then generates (k-1)-itemsets candidates from them → checks their support → keeps frequent ones →
break if found frequent itselt at (k-1)-itemset is 0
....

So when it finishes, you have:

All frequent 1-itemsets

All frequent 2-itemsets

All frequent 3-itemsets

... up to the largest k-itemset (say, 3-itemsets in your example)

✅ This means all smaller itemsets (like {A, B}) that met the minimum support are already known before reaching {A, B, C}.

Phase 2 — Generate association rules
======================================================

Now, for each frequent itemset (of any size), Apriori generates rules of the form:

𝑋 ⇒ 𝑌

such that X∪Y is the frequent itemset and X∩Y=∅.

So for example, if {A, B, C} is frequent:

You generate rules like:

{A, B} → {C}

{A, C} → {B}

{B, C} → {A}

{A} → {B, C}

{B} → {A, C}

{C} → {A, B}

And independently, rules are also generated from smaller frequent sets like {A, B}:

{A} → {B}

{B} → {A}



Phase-3 (Find qualified rules based on confidence)
==================================================================================
    Then check confidence of each possible rule (s1 to sn)
    [Each rule of s-i signifies how subset s-i influences the rest(I- s-i) of the elements of the itemset(I)]
        Rule-i = s-i -> (I - s-i)
    If confidence of Rule-i is > minimum Confidence value (MINCONF) rhen
        Qualify the rule as valid rule

