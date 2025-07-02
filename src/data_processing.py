# Feature engineering and data processing functions will go here.
# import pandas as pd
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# --- Utility function for aggregation ---
def most_frequent(series):
    """Return the most frequent value in a Series (mode)."""
    return series.mode()[0] if not series.mode().empty else np.nan

# --- A. Aggregate Features Transformer ---
class AggregateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data at the customer level, including numerical and categorical features.
    - Numerical: total, average, count, std of Amount
    - Categorical: most frequent ProductCategory, ChannelId, ProviderId
    - Date/Time: most frequent transaction hour, month, year
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount',
                 cat_cols=['ProductCategory', 'ChannelId', 'ProviderId'],
                 datetime_col='TransactionStartTime'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.cat_cols = cat_cols
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convert datetime column
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        # Aggregation dictionary
        agg_dict = {
            self.amount_col: ['sum', 'mean', 'count', 'std'],
            'transaction_hour': most_frequent,
            'transaction_month': most_frequent,
            'transaction_year': most_frequent
        }
        for col in self.cat_cols:
            agg_dict[col] = most_frequent
        agg_df = X.groupby(self.customer_id_col).agg(agg_dict)
        # Flatten MultiIndex columns
        agg_df.columns = ['_'.join([c for c in col if c]) for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()
        return agg_df

# --- B. Feature Engineering Pipeline Builder ---
def build_feature_engineering_pipeline(scaling='standard', use_woe=False):
    """
    Build a robust sklearn pipeline for feature engineering.
    Args:
        scaling: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        use_woe: If True, apply WoE encoding to categorical features (placeholder)
    Returns:
        sklearn Pipeline object
    """
    # Columns after aggregation
    categorical_cols = [
        'ProductCategory_most_frequent',
        'ChannelId_most_frequent',
        'ProviderId_most_frequent'
    ]
    numerical_cols = [
        'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
        'transaction_hour_most_frequent',
        'transaction_month_most_frequent',
        'transaction_year_most_frequent'
    ]
    # Imputation and scaling pipeline for numerical features
    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler() if scaling == 'standard' else MinMaxScaler())
    ])
    # Categorical encoding pipeline
    cat_pipeline = Pipeline([
        # Placeholder for WoE encoding
        # ('woe', WoEEncoder()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # ColumnTransformer to apply transformations
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='passthrough')
    # Full pipeline
    pipeline = Pipeline([
        ('aggregate_features', AggregateFeaturesTransformer()),
        ('preprocessing', preprocessor)
    ])
    return pipeline

# --- Example usage (for testing, not for production) ---
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('../data/raw/data.csv')
    # Build and fit the pipeline (choose scaling: 'standard' or 'minmax')
    pipeline = build_feature_engineering_pipeline(scaling='standard')
    processed = pipeline.fit_transform(df)
    print(processed)
