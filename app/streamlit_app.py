import os
import json
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_FEATURES_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'customer_features.csv')
PREDICTIONS_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'predictions.csv')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')
SHAP_SUMMARY_PATH = os.path.join(MODELS_DIR, 'shap_summary.png')

st.set_page_config(page_title='SmartLoanScorer Dashboard', layout='wide')
st.title('SmartLoanScorer: Credit Risk Scoring')
st.caption('Transparent, auditable, and production-ready credit scoring for BNPL lending')

col1, col2 = st.columns(2)

with col1:
    st.subheader('1) Upload customer features CSV')
    uploaded = st.file_uploader('CSV with engineered features', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif os.path.exists(DEFAULT_FEATURES_CSV):
        df = pd.read_csv(DEFAULT_FEATURES_CSV)
        st.info('Loaded default processed features.')
    else:
        df = None
    if df is not None:
        st.dataframe(df.head(20))

with col2:
    st.subheader('2) Model and Explainability')
    if os.path.exists(BEST_MODEL_PATH):
        st.success('Best model found. Ready to score.')
    else:
        st.warning('Best model not found. Train the model first.')
    if os.path.exists(SHAP_SUMMARY_PATH):
        st.image(SHAP_SUMMARY_PATH, caption='Global Feature Importance (SHAP summary)')

st.subheader('3) Score customers')
if st.button('Run scoring'):
    if df is None:
        st.error('Please provide a features CSV.')
    elif not os.path.exists(BEST_MODEL_PATH):
        st.error('Model not found. Please run training.')
    else:
        model = joblib.load(BEST_MODEL_PATH)
        feature_cols = [c for c in df.columns if c not in ['CustomerId', 'is_high_risk']]
        proba = model.predict_proba(df[feature_cols])[:, 1] if hasattr(model, 'predict_proba') else model.predict(df[feature_cols])
        out = df[['CustomerId']].copy() if 'CustomerId' in df.columns else pd.DataFrame({'row_id': range(len(df))})
        out['prob_default'] = proba
        out.to_csv(PREDICTIONS_CSV, index=False)
        st.success('Scoring complete. Preview below:')
        st.dataframe(out.head(50))
        st.download_button('Download predictions CSV', data=out.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

st.markdown('---')
st.caption('Basel II-aligned: model transparency, monitoring, reproducibility, and auditability built-in.') 