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
├── app/
│   └── streamlit_app.py       # Streamlit dashboard for demo
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## What’s new (finance-grade improvements)

- Persisted best model to `models/best_model.joblib` for reliable deployment and audit.
- Logged global explainability artifact `models/shap_summary.png` using SHAP.
- Inference: robust CLI `src/predict.py` for batch scoring CSVs.
- API: MLflow registry load with automatic local model fallback.
- Dashboard: `app/streamlit_app.py` to demo scoring and show explainability.
- CI/CD: basic API sanity test added; requirements extended for coverage and httpx.

## Quickstart

- Prepare data features and target as per pipeline (see below), then train:

```bash
python -m src.train
```

Artifacts saved to `models/` and logged in MLflow at `mlruns/`.

- Run API locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Batch predict on processed features:

```bash
python -m src.predict --input_csv data/processed/customer_features.csv --output_csv data/processed/predictions.csv
```

- Run the dashboard:

```bash
streamlit run app/streamlit_app.py
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

# Model Training and Tracking (Task 5)

As part of Task 5, a structured and reproducible model training process was implemented, including experiment tracking, model versioning, and unit testing.

**Key steps in the model training and tracking pipeline:**

- **Data Splitting:**

  - The processed customer-level dataset (with proxy target) is split into training and testing sets to evaluate model performance on unseen data.

- **Model Selection and Training:**

  - At least two models are trained: Logistic Regression and Random Forest.
  - Hyperparameter tuning is performed using GridSearchCV for both models.

- **Model Evaluation:**

  - Models are evaluated using multiple metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
  - The best model is selected based on ROC-AUC score.

- **Experiment Tracking with MLflow:**

  - All experiments, parameters, metrics, and trained models are tracked using MLflow.
  - The best model is registered in the MLflow Model Registry for versioning and reproducibility.
  - The MLflow tracking server is configured to use a local `mlruns/` directory (which should be added to `.gitignore`).

- **Unit Testing:**
  - Unit tests for data processing helper functions (e.g., `most_frequent`) are implemented in `tests/test_data_processing.py`.
  - All tests pass, ensuring the reliability of the data processing pipeline.

## Task 6 - Model Deployment and Continuous Integration

In this final task, the trained credit risk model is deployed as a REST API using FastAPI. The API loads the best model from the MLflow Model Registry and exposes a `/predict` endpoint for batch risk scoring. Pydantic models are used for request and response validation to ensure robust data handling.

The project is containerized using Docker, allowing for easy deployment and scalability. A `docker-compose.yml` file is provided for streamlined local development and deployment.

Continuous Integration (CI) is set up using GitHub Actions. The workflow automatically runs code linting (with flake8) and unit tests (with pytest) on every push or pull request to the main branch. This ensures code quality and reliability before deployment.

**Key Deliverables:**

- FastAPI application for serving predictions (`src/api/main.py`)
- Pydantic models for request/response validation (`src/api/pydantic_models.py`)
- Dockerfile and docker-compose for containerization
- `.github/workflows/ci.yml` for automated linting and testing

**How to use:**

- Run the API locally or with Docker to serve predictions.
- Use the `/predict` endpoint to get risk probabilities for new customers.
- All code changes are automatically checked for style and correctness via CI.

This task completes the end-to-end MLOps workflow, making the credit risk model production-ready
