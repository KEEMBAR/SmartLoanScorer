from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse, FEATURES
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="SmartLoanScorer API", description="API for customer risk prediction using MLflow model registry.")

# --- Load the best model from MLflow Model Registry ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:../mlruns")
MODEL_NAME = "customer_risk_model"
MODEL_STAGE = "None"  # Use 'None' to always get the latest version

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    # Get the latest version (or you can specify a stage like 'Production')
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    model = None
    print(f"Error loading model from MLflow: {e}")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    # Convert input to DataFrame
    try:
        data = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(data, columns=FEATURES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")
    # Predict probabilities
    try:
        preds = model.predict(df)
        # If model returns probabilities for both classes, take the positive class
        if hasattr(preds, 'shape') and len(preds.shape) > 1 and preds.shape[1] == 2:
            probabilities = preds[:, 1].tolist()
        elif hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[:, 1].tolist()
        else:
            # If model returns only one value per sample
            probabilities = preds.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return PredictResponse(probabilities=probabilities)

@app.get("/")
def root():
    return {"message": "SmartLoanScorer API is running. Use /predict to get risk probabilities."} 