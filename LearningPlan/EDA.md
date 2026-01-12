EDA and Processing are done in two phases. Some are done before the train-test split, while others are done after. 
This distinction is crucial to avoid data leakage and ensure that the model generalizes well to unseen data.

# **EDA & Preprocessing — What Happens Before vs After Train-Test Split**

| Step                              | Activity                                                    | Before Split | After Split           | Notes / Purpose           |
|-----------------------------------|-------------------------------------------------------------| ------------ | --------------------- |---------------------------|
| **Data loading**                  | Read dataset                                                | ✅            | —                     | Load raw data             |
| **Initial inspection**            | Shape, head(), info(), describe()                           | ✅            | —                     | Understand structure      |
| **Data type correction**          | Convert object → numeric/date                               | ✅            | —                     | Structural cleanup        |
| **Duplicate removal**             | Drop duplicate rows                                         | ✅            | —                     | No learned statistics     |
| **Obvious data errors**           | Fix impossible values (e.g., negative age)                  | ✅            | —                     | Rule-based fixes          |
| **Replace sentinel values**       | Replace special codes (e.g., 0 → NaN if zero means missing) | ✅            | —                     | Structural transformation |
| **Drop high-missing columns**     | Drop columns with >X% nulls                                 | ✅            | —                     | No statistical inference  |
| **Univariate EDA**                | Histograms, boxplots, value counts                          | ✅            | —                     | Understand distributions  |
| **Correlation Analysis**          | Visually inspect correlation between independent variables  | ✅            | —                     | Understand correlation    |
| **Target leakage check**          | Remove columns that leak target                             | ✅            | —                     | Critical sanity step      |
| **Train-test split**              | Create X_train / X_test                                     | —            | ✅                     | First ML boundary         |
| **Missing value imputation**      | Mean/median/mode/KNN imputation                             | ❌            | ✅ (fit on train only) | Prevent leakage           |
| **Outlier capping**               | Winsorization / clipping based on percentiles               | ❌            | ✅ (fit on train only) | Uses learned thresholds   |
| **Scaling / normalization**       | StandardScaler / MinMaxScaler                               | ❌            | ✅ (fit on train only) | Uses learned stats        |
| **Encoding categorical features** | OneHot / Ordinal encoding                                   | ❌            | ✅ (fit on train only) | Learns category set       |
| **Feature selection**             | Correlation filtering / model-based                         | ❌            | ✅                     | Must use train only       |
| **Dimensionality reduction**      | PCA / t-SNE / UMAP                                          | ❌            | ✅ (fit on train only) | Learns projections        |
| **Model training**                | Fit ML model                                                | ❌            | ✅                     | Train phase               |
| **Evaluation**                    | Metrics on test set                                         | ❌            | ✅                     | Honest performance        |

---

## **Simple guiding principle**

**Before split → rule-based cleaning & exploration**
**After split → anything that learns from data**

---

