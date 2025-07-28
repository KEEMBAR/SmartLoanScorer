"""
Unit tests for data processing functions
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from src.data_processing import most_frequent, AggregateFeaturesTransformer

class TestMostFrequent:
    """Test cases for the most_frequent function."""
    
    def test_most_frequent_with_single_mode(self):
        """Test most_frequent with a series that has one clear mode."""
        series = pd.Series([1, 2, 2, 3, 2, 4])
        result = most_frequent(series)
        assert result == 2
    
    def test_most_frequent_with_multiple_modes(self):
        """Test most_frequent with a series that has multiple modes."""
        series = pd.Series([1, 2, 2, 3, 3, 4])
        result = most_frequent(series)
        # Should return the first mode when there are multiple
        assert result in [2, 3]
    
    def test_most_frequent_with_empty_series(self):
        """Test most_frequent with an empty series."""
        series = pd.Series([])
        result = most_frequent(series)
        assert np.isnan(result)
    
    def test_most_frequent_with_all_nan(self):
        """Test most_frequent with a series containing only NaN values."""
        series = pd.Series([np.nan, np.nan, np.nan])
        result = most_frequent(series)
        assert np.isnan(result)
    
    def test_most_frequent_with_mixed_nan(self):
        """Test most_frequent with a series containing NaN and regular values."""
        series = pd.Series([1, 2, np.nan, 2, 3, np.nan])
        result = most_frequent(series)
        assert result == 2

class TestAggregateFeaturesTransformer:
    """Test cases for the AggregateFeaturesTransformer class."""
    
    def test_aggregate_features_transformer_initialization(self):
        """Test that the transformer initializes with default parameters."""
        transformer = AggregateFeaturesTransformer()
        assert transformer.customer_id_col == 'CustomerId'
        assert transformer.amount_col == 'Amount'
        assert transformer.cat_cols == ['ProductCategory', 'ChannelId', 'ProviderId']
        assert transformer.datetime_col == 'TransactionStartTime'
    
    def test_aggregate_features_transformer_custom_initialization(self):
        """Test that the transformer initializes with custom parameters."""
        transformer = AggregateFeaturesTransformer(
            customer_id_col='CustomerID',
            amount_col='TransactionAmount',
            cat_cols=['Category', 'Channel'],
            datetime_col='TransactionDate'
        )
        assert transformer.customer_id_col == 'CustomerID'
        assert transformer.amount_col == 'TransactionAmount'
        assert transformer.cat_cols == ['Category', 'Channel']
        assert transformer.datetime_col == 'TransactionDate'
    
    def test_fit_method(self):
        """Test that the fit method returns self."""
        transformer = AggregateFeaturesTransformer()
        X = pd.DataFrame({'CustomerId': [1, 2], 'Amount': [100, 200]})
        result = transformer.fit(X)
        assert result is transformer
    
    def test_transform_method_basic(self):
        """Test the transform method with basic data."""
        transformer = AggregateFeaturesTransformer()
        
        # Create sample data
        data = {
            'CustomerId': ['C1', 'C1', 'C2', 'C2'],
            'Amount': [100, 200, 150, 250],
            'ProductCategory': ['A', 'A', 'B', 'B'],
            'ChannelId': ['CH1', 'CH1', 'CH2', 'CH2'],
            'ProviderId': ['P1', 'P1', 'P2', 'P2'],
            'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00',
                                   '2023-01-01 12:00:00', '2023-01-01 13:00:00']
        }
        X = pd.DataFrame(data)
        
        # Transform
        result = transformer.transform(X)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that we have one row per customer
        assert len(result) == 2
        
        # Check that CustomerId column is present
        assert 'CustomerId' in result.columns
        
        # Check that aggregated columns are present
        expected_agg_cols = ['Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
                           'transaction_hour_most_frequent', 'transaction_month_most_frequent',
                           'transaction_year_most_frequent', 'ProductCategory_most_frequent',
                           'ChannelId_most_frequent', 'ProviderId_most_frequent']
        
        for col in expected_agg_cols:
            assert col in result.columns, f"Expected column {col} not found in result"
    
    def test_transform_method_with_empty_data(self):
        """Test the transform method with empty data."""
        transformer = AggregateFeaturesTransformer()
        X = pd.DataFrame(columns=['CustomerId', 'Amount', 'ProductCategory', 
                                 'ChannelId', 'ProviderId', 'TransactionStartTime'])
        
        result = transformer.transform(X)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

if __name__ == "__main__":
    pytest.main([__file__])

