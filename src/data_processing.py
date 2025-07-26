import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime
 
# ---------- Utility ----------
def most_frequent(series):
    return series.mode()[0] if not series.mode().empty else np.nan

# ---------- Custom Transformers ----------

class AggregateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data at the customer level, including:
    - Numerical: total, average, count, std of Amount
    - Categorical: most frequent ProductCategory, ChannelId, ProviderId
    - Datetime: most frequent hour, day, month, year
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
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year

        agg_dict = {
            self.amount_col: ['sum', 'mean', 'count', 'std'],
            'transaction_hour': most_frequent,
            'transaction_day': most_frequent,
            'transaction_month': most_frequent,
            'transaction_year': most_frequent
        }

        for col in self.cat_cols:
            agg_dict[col] = most_frequent

        agg_df = X.groupby(self.customer_id_col).agg(agg_dict)
        agg_df.columns = ['_'.join([str(c) for c in col if c]) for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        return agg_df

# ---------- Feature Engineering Pipeline ----------

def build_feature_engineering_pipeline(encoding='onehot', scaling='standard'):
    """
    encoding: 'onehot' or 'label'
    scaling: 'standard' or 'minmax'
    """
    categorical_cols = [
        'ProductCategory_most_frequent',
        'ChannelId_most_frequent',
        'ProviderId_most_frequent'
    ]
    numerical_cols = [
        'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
        'transaction_hour_most_frequent',
        'transaction_day_most_frequent',
        'transaction_month_most_frequent',
        'transaction_year_most_frequent'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler() if scaling == 'standard' else MinMaxScaler())
    ])

    if encoding == 'onehot':
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    elif encoding == 'label':
        # Label encoding per column
        class MultiLabelEncoder(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                self.encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.columns}
                return self

            def transform(self, X):
                return pd.DataFrame({
                    col: self.encoders[col].transform(X[col].astype(str))
                    for col in X.columns
                })
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('label', MultiLabelEncoder())
        ])
    else:
        raise ValueError(f"Unsupported encoding method: {encoding}")

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='passthrough')  # Keeps CustomerId

    pipeline = Pipeline([
        ('aggregate_features', AggregateFeaturesTransformer()),
        ('preprocessing', preprocessor)
    ])

    return pipeline

# ---------- Execution Example ----------

if __name__ == "__main__":
    # 1. Load raw data
    df = pd.read_csv('data/raw/data.csv')

    # 2. Build pipeline
    pipeline = build_feature_engineering_pipeline(encoding='onehot', scaling='standard')

    # 3. Fit & transform
    processed_array = pipeline.fit_transform(df)

    # 4. Generate final column names
    preprocessor = pipeline.named_steps['preprocessing']
    ohe = None
    if 'onehot' in preprocessor.named_transformers_['cat'].named_steps:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_cols = preprocessor.transformers_[1][2]
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    else:
        ohe_feature_names = preprocessor.transformers_[1][2]

    num_cols = preprocessor.transformers_[0][2]
    passthrough = ['CustomerId']  # assumed passed through
    final_feature_names = list(num_cols) + list(ohe_feature_names) + passthrough

    # 5. Save processed features
    processed_df = pd.DataFrame(processed_array, columns=final_feature_names)
    processed_df.to_csv('data/processed/customer_features.csv', index=False)
    print("✅ Processed features saved to data/processed/customer_features.csv")
