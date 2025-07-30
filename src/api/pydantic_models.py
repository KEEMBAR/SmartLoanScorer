from pydantic import BaseModel, Field
from typing import List, Optional

# List of features expected by the model (from the processed CSV, excluding CustomerId and is_high_risk)
FEATURES = [
    'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
    'transaction_hour_most_frequent', 'transaction_day_most_frequent',
    'transaction_month_most_frequent', 'transaction_year_most_frequent',
    'ProductCategory_most_frequent_airtime',
    'ProductCategory_most_frequent_data_bundles',
    'ProductCategory_most_frequent_financial_services',
    'ProductCategory_most_frequent_movies',
    'ProductCategory_most_frequent_other',
    'ProductCategory_most_frequent_ticket',
    'ProductCategory_most_frequent_transport',
    'ProductCategory_most_frequent_tv',
    'ProductCategory_most_frequent_utility_bill',
    'ChannelId_most_frequent_ChannelId_1',
    'ChannelId_most_frequent_ChannelId_2',
    'ChannelId_most_frequent_ChannelId_3',
    'ChannelId_most_frequent_ChannelId_5',
    'ProviderId_most_frequent_ProviderId_1',
    'ProviderId_most_frequent_ProviderId_2',
    'ProviderId_most_frequent_ProviderId_3',
    'ProviderId_most_frequent_ProviderId_4',
    'ProviderId_most_frequent_ProviderId_5',
    'ProviderId_most_frequent_ProviderId_6',
    'Recency', 'Frequency', 'Monetary'
]

class CustomerFeatures(BaseModel):
    """Request model for a single customer."""
    Amount_sum: float
    Amount_mean: float
    Amount_count: float
    Amount_std: float
    transaction_hour_most_frequent: float
    transaction_day_most_frequent: float
    transaction_month_most_frequent: float
    transaction_year_most_frequent: float
    ProductCategory_most_frequent_airtime: float
    ProductCategory_most_frequent_data_bundles: float
    ProductCategory_most_frequent_financial_services: float
    ProductCategory_most_frequent_movies: float
    ProductCategory_most_frequent_other: float
    ProductCategory_most_frequent_ticket: float
    ProductCategory_most_frequent_transport: float
    ProductCategory_most_frequent_tv: float
    ProductCategory_most_frequent_utility_bill: float
    ChannelId_most_frequent_ChannelId_1: float
    ChannelId_most_frequent_ChannelId_2: float
    ChannelId_most_frequent_ChannelId_3: float
    ChannelId_most_frequent_ChannelId_5: float
    ProviderId_most_frequent_ProviderId_1: float
    ProviderId_most_frequent_ProviderId_2: float
    ProviderId_most_frequent_ProviderId_3: float
    ProviderId_most_frequent_ProviderId_4: float
    ProviderId_most_frequent_ProviderId_5: float
    ProviderId_most_frequent_ProviderId_6: float
    Recency: float
    Frequency: float
    Monetary: float

class PredictRequest(BaseModel):
    """Request model for batch prediction."""
    customers: List[CustomerFeatures]

class PredictResponse(BaseModel):
    """Response model for prediction results."""
    probabilities: List[float] 
    