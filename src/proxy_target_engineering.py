"""
Proxy Target Engineering Script

- Loads processed customer features
- Computes RFM (Recency, Frequency, Monetary) metrics
- Clusters customers using KMeans (number of clusters is parameterized)
- Assigns a binary 'is_high_risk' label to the cluster with highest Recency, lowest Frequency, and lowest Monetary
- Saves the result with the new target variable for downstream modeling
- Robust to being run from any directory
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Parameters ---
N_CLUSTERS = 3  

# --- Path setup: robust to being run from any directory ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(BASE_DIR, 'data', 'processed', 'customer_features.csv')
output_path = os.path.join(BASE_DIR, 'data', 'processed', 'customer_features_with_target.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- Load processed customer features ---
customer_df = pd.read_csv(input_path)

# --- Calculate RFM metrics ---
# Recency: Uses 'last_transaction_date' if available; otherwise, uses the most recent year as a proxy
if 'last_transaction_date' in customer_df.columns:
    snapshot_date = pd.to_datetime(customer_df['last_transaction_date']).max() + pd.Timedelta(days=1)
    customer_df['Recency'] = (snapshot_date - pd.to_datetime(customer_df['last_transaction_date'])).dt.days
else:
    customer_df['Recency'] = customer_df['transaction_year_most_frequent'].max() - customer_df['transaction_year_most_frequent']

# Frequency: Number of transactions per customer
customer_df['Frequency'] = customer_df['Amount_count']

# Monetary: Total transaction amount per customer
customer_df['Monetary'] = customer_df['Amount_sum']

# --- Prepare RFM DataFrame for clustering ---
rfm_df = customer_df[['CustomerId', 'Recency', 'Frequency', 'Monetary']].copy()

# --- Scale RFM features ---
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

# --- Cluster customers using KMeans ---
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)

# --- Identify the high-risk cluster ---
cluster_summary = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
high_risk_cluster = cluster_summary.sort_values(['Frequency', 'Monetary', 'Recency'], ascending=[True, True, False]).index[0]
rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)

# --- Merge the high-risk label back to the customer DataFrame ---
customer_df = customer_df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# --- Save the final DataFrame with the new target variable ---
customer_df.to_csv(output_path, index=False)

print(f'✅ Proxy target variable (is_high_risk) created and merged. Saved to {output_path}')
