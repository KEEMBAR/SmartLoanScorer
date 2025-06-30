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
