import argparse
import os
import pandas as pd
import joblib

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'customer_features.csv')
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'predictions.csv')


def find_latest_model(models_dir: str) -> str:
    candidates = []
    for root, _, files in os.walk(models_dir):
        for f in files:
            if f.endswith('.joblib'):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No .joblib model files found in {models_dir}")
    return max(candidates, key=os.path.getmtime)


essential_exclude = {'CustomerId', 'is_high_risk'}


def run_inference(model_path: str, input_csv: str, output_csv: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c not in essential_exclude]
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(df[feature_cols])[:, 1]
    else:
        preds = model.predict(df[feature_cols])
        probs = preds if getattr(preds, 'ndim', 1) == 1 else preds[:, 1]
    out = df[['CustomerId']].copy() if 'CustomerId' in df.columns else pd.DataFrame()
    out['prob_default'] = probs
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Batch predict customer risk probabilities')
    parser.add_argument('--models_dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--input_csv', default=DEFAULT_INPUT)
    parser.add_argument('--output_csv', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model(os.path.abspath(args.models_dir))
    run_inference(model_path, os.path.abspath(args.input_csv), os.path.abspath(args.output_csv))


if __name__ == '__main__':
    main() 