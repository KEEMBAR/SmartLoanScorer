"""
Model Training and Tracking Script

This script:
1. Loads processed customer features with target variable
2. Splits data into training and testing sets
3. Trains multiple models (Logistic Regression, Random Forest)
4. Performs hyperparameter tuning using GridSearchCV
5. Evaluates models using multiple metrics
6. Tracks experiments and registers the best model with MLflow
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
from datetime import datetime
import json
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --- Path setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'customer_features_with_target.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')
SHAP_SUMMARY_PATH = os.path.join(MODELS_DIR, 'shap_summary.png')

# --- MLflow setup ---
mlflow.set_tracking_uri("file:" + os.path.join(BASE_DIR, "mlruns"))
mlflow.set_experiment("customer_risk_prediction")

def save_local_model_and_artifacts(model, X_train: pd.DataFrame):
    """Persist the best model, feature names, and SHAP summary plot to models/.
    """
    # Save model
    joblib.dump(model, BEST_MODEL_PATH)
    # Save feature names
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(list(X_train.columns), f)
    # Compute and save SHAP summary
    try:
        sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)
        explainer = None
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, sample)
        else:
            explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PATH)
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")

def load_and_prepare_data():
    """Load and prepare data for training."""
    print("📊 Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    target_col = 'is_high_risk'
    feature_cols = [col for col in df.columns if col not in ['CustomerId', target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📈 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model using multiple metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n📊 {model_name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with hyperparameter tuning."""
    print("\n🔍 Training Logistic Regression...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    # Grid search
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    print("\n🌲 Training Random Forest...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    """Main training pipeline."""
    print("🚀 Starting Model Training and Tracking Pipeline")
    print("=" * 50)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    models = {}
    
    # Logistic Regression
    with mlflow.start_run(run_name="logistic_regression"):
        lr_model = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        
        # Log parameters and metrics
        mlflow.log_params(lr_model.get_params())
        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(lr_model, "logistic_regression")
        
        models['logistic_regression'] = {
            'model': lr_model,
            'metrics': lr_metrics
        }
    
    # Random Forest
    with mlflow.start_run(run_name="random_forest"):
        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Log parameters and metrics
        mlflow.log_params(rf_model.get_params())
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, "random_forest")
        
        models['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics
        }
    
    # Find best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['metrics']['roc_auc'])
    best_model = models[best_model_name]['model']
    best_metrics = models[best_model_name]['metrics']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"🏆 Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # Save locally and log artifacts
    save_local_model_and_artifacts(best_model, X_train)
    print(f"💾 Best model saved to {BEST_MODEL_PATH}")

    # Register best model
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_model.get_params())
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, "best_model")
        # Log local artifacts to MLflow as well (optional)
        if os.path.exists(BEST_MODEL_PATH):
            mlflow.log_artifact(BEST_MODEL_PATH, artifact_path="artifacts")
        if os.path.exists(SHAP_SUMMARY_PATH):
            mlflow.log_artifact(SHAP_SUMMARY_PATH, artifact_path="artifacts")
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
        mlflow.register_model(model_uri, "customer_risk_model")
    
    print(f"\n✅ Training completed! Best model registered in MLflow Model Registry")
    print(f"📁 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    return best_model, best_metrics

if __name__ == "__main__":
    best_model, best_metrics = main() 