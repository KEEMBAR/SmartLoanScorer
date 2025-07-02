# SmartLoanScorer

SmartLoanScorer An ML‑powered credit scoring engine that converts customer RFM behavior into interpretable risk scores for BNPL lending—fully transparent and Basel II compliant.

# Project Structure

The project follows a standardized structure as mandated by the challenge:

```
SmartLoanScorer/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

# Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord emphasizes rigorous risk measurement and management for financial institutions, requiring them to quantify and justify their credit risk exposures. This regulatory framework demands that credit scoring models be not only accurate but also interpretable and well-documented. An interpretable model allows risk managers, auditors, and regulators to understand how credit decisions are made, ensuring transparency, accountability, and compliance. Well-documented models facilitate validation, monitoring, and updates, which are essential for regulatory approval and for maintaining trust with stakeholders.

### 2. Necessity and Risks of Proxy Variables

In the absence of a direct "default" label in the dataset, creating a proxy variable (such as identifying high-risk customers based on behavioral patterns like RFM—Recency, Frequency, Monetary value) is necessary to enable supervised learning. However, relying on a proxy introduces potential business risks:

- Misalignment with true default risk: The proxy may not perfectly capture actual default behavior, leading to misclassification.
- Regulatory scrutiny: Models based on proxies may be challenged by regulators if the proxy is not well-justified or transparent.
- Business impact: Incorrect risk predictions can result in either lost revenue (by rejecting good customers) or increased losses (by approving high-risk customers).

Careful design, validation, and documentation of the proxy are essential to mitigate these risks.

### 3. Trade-offs: Simple vs. Complex Models

- **Simple, Interpretable Models (e.g., Logistic Regression with WoE):**
  - Pros: High transparency, easy to explain to regulators, straightforward to validate and monitor, lower risk of overfitting.
  - Cons: May have lower predictive performance, especially with complex or non-linear relationships in the data.
- **Complex, High-Performance Models (e.g., Gradient Boosting):**
  - Pros: Often achieve higher accuracy and better capture complex patterns.
  - Cons: Harder to interpret, more challenging to document and justify to regulators, and may require additional tools (e.g., SHAP values) for explanation.

In regulated financial contexts, the choice often balances predictive power with the need for interpretability and regulatory compliance.

# Exploratory Data Analysis (EDA)

As part of Task 2, an in-depth exploratory data analysis (EDA) was conducted to understand the structure, quality, and patterns in the dataset. The following steps were performed:

1. **Overview of the Data**: Loaded the dataset, checked the number of rows and columns, and previewed the first few records.
2. **Data Types and Basic Info**: Inspected data types and non-null counts for each column.
3. **Summary Statistics**: Generated summary statistics for both numerical and categorical features.
4. **Distribution of Numerical Features**: Visualized distributions using histograms to identify patterns, skewness, and outliers.
5. **Distribution of Categorical Features**: Analyzed the frequency of categories using bar plots.
6. **Correlation Analysis**: Computed and visualized the correlation matrix for numerical features.
7. **Identifying Missing Values**: Checked for missing values and visualized their presence using a heatmap.
8. **Outlier Detection**: Used box plots to identify outliers in key numerical features.

## Key EDA Insights

- **No Missing Values**: All columns are complete; no imputation or row removal is needed.
- **High Correlation**: `Amount` and `Value` are extremely highly correlated (≈ 0.99), suggesting redundancy.
- **FraudResult Correlation**: `FraudResult` has a moderate positive correlation with both `Amount` and `Value` (≈ 0.56–0.57).
- **Categorical Features**: `CurrencyCode` has only one unique value and can be dropped. Features like `ProviderId`, `ProductId`, `ProductCategory`, and `ChannelId` are suitable for encoding. High-cardinality columns (`AccountId`, `SubscriptionId`, `CustomerId`) are better used for aggregation (e.g., RFM features).
- **Data Uniqueness**: `TransactionId` is unique for every row; repeated values in customer/account columns confirm multiple transactions per customer.
- **No Major Data Quality Issues**: Data types and value ranges are as expected; no significant outliers or anomalies detected.

# Feature Engineering (Task 3)

As part of Task 3, a robust, automated, and reproducible feature engineering pipeline was developed and implemented in `src/data_processing.py`. This pipeline prepares the raw transaction data for model training by transforming it into meaningful, model-ready features at the customer level. The pipeline is built using scikit-learn's `Pipeline` and `ColumnTransformer` for modularity and reproducibility.

**Key steps in the feature engineering pipeline:**

- **Aggregation:**

  - Numerical features: For each customer, the pipeline computes the total, average, count, and standard deviation of transaction amounts.
  - Categorical features: For each customer, the most frequent `ProductCategory`, `ChannelId`, and `ProviderId` are determined.
  - Date/Time features: The most frequent transaction hour, month, and year are extracted per customer.

- **Categorical Encoding:**

  - Aggregated categorical features are encoded using OneHotEncoder, converting them into binary vectors suitable for machine learning models.

- **Numerical Feature Scaling:**

  - Numerical features are imputed for missing values (if any) and then scaled using either standardization (StandardScaler) or normalization (MinMaxScaler), as selected in the pipeline.

- **Pipeline Automation:**
  - All steps are chained together using scikit-learn's `Pipeline` and `ColumnTransformer`, ensuring that the same transformations are applied consistently during both training and inference.

This feature engineering approach ensures that the data is clean, consistent, and contains rich information about each customer's transaction behavior, providing a strong foundation for building predictive credit risk models.
