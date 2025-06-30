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
