from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse, FEATURES
import mlflow.pyfunc
import pandas as pd
import os
import joblib

app = FastAPI(title="SmartLoanScorer API", description="API for customer risk prediction using MLflow model registry.")

# --- Load the best model from MLflow Model Registry ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{os.path.join(BASE_DIR, 'mlruns')}")
MODEL_NAME = "customer_risk_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None

try:
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    # Fallback to local joblib model
    try:
        local_model_path = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
        if os.path.exists(local_model_path):
            model = joblib.load(local_model_path)
            print(f"Loaded local model from {local_model_path}")
    except Exception as e2:
        print(f"Error loading local model: {e2}")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        data = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(data, columns=FEATURES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")
    try:
        if hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
            preds = model.predict(df)
            probabilities = preds.tolist() if getattr(preds, 'ndim', 1) == 1 else preds[:, 1].tolist()
        else:
            probabilities = model.predict(df).tolist() if isinstance(model, mlflow.pyfunc.PyFuncModel) else model.predict_proba(df)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return PredictResponse(probabilities=probabilities)

@app.get("/")
def root():
    return {"message": "SmartLoanScorer API is running. Use /predict to get risk probabilities."} 