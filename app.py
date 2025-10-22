# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import PowerTransformer

# -----------------------------
# Configuration
# -----------------------------
WORK_DIR = Path.cwd()
MODELS_PATH = WORK_DIR / "models.pkl"   # dictionary of models
SCALER_PATH = WORK_DIR / "scaler.pkl"   # optional scaler if used

FEATURE_NAMES = [
    'Happiness Rank', 'Standard Error', 'Economy (GDP per Capita)', 'Family',
    'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
    'Generosity', 'Dystopia Residual'
]

st.set_page_config(page_title="World Happiness Prediction", layout="centered")
st.title("🌎 World Happiness Score Prediction")
st.write("Enter the feature values below and see predictions from available regression models.")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    models = load_pickle(MODELS_PATH)
except Exception as e:
    st.error(f"Error loading models: {e}")
    models = None

try:
    scaler = load_pickle(SCALER_PATH)
except Exception as e:
    st.warning(f"Scaler not loaded or not found: {e}")
    scaler = None

# -----------------------------
# User input
# -----------------------------
def user_input_features():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        happiness_rank = st.number_input("Happiness Rank", min_value=1, max_value=150, value=50)
        standard_error = st.number_input("Standard Error", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
        gdp_per_capita = st.number_input("Economy (GDP per Capita)", min_value=0.0, max_value=5.0, value=1.0, step=0.01)
    with col2:
        family = st.number_input("Family", min_value=0.0, max_value=5.0, value=1.0, step=0.01)
        health = st.number_input("Health (Life Expectancy)", min_value=0.0, max_value=5.0, value=1.0, step=0.01)
        freedom = st.number_input("Freedom", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    with col3:
        trust = st.number_input("Trust (Government Corruption)", min_value=0.0, max_value=5.0, value=0.1, step=0.01)
        generosity = st.number_input("Generosity", min_value=0.0, max_value=5.0, value=0.1, step=0.01)
        dystopia_residual = st.number_input("Dystopia Residual", min_value=0.0, max_value=5.0, value=1.0, step=0.01)

    data = {
        'Happiness Rank': happiness_rank,
        'Standard Error': standard_error,
        'Economy (GDP per Capita)': gdp_per_capita,
        'Family': family,
        'Health (Life Expectancy)': health,
        'Freedom': freedom,
        'Trust (Government Corruption)': trust,
        'Generosity': generosity,
        'Dystopia Residual': dystopia_residual
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

with st.expander("Input features (preview)"):
    st.write(input_df)

# -----------------------------
# Power Transform features
# -----------------------------
pt = PowerTransformer()
try:
    # Fit only if no scaler saved, otherwise use scaler
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = pt.fit_transform(input_df)  # transform user input
except Exception as e:
    st.error(f"Error transforming input features: {e}")
    st.stop()

# -----------------------------
# Predictions
# -----------------------------
if st.button("Predict Happiness Score"):
    if models is None:
        st.error("No models loaded. Check models.pkl file.")
    else:
        results = {}
        for name, model in models.items():
            try:
                pred = model.predict(input_scaled)
                results[name] = float(pred.ravel()[0])
            except Exception as e:
                results[name] = f"Error: {e}"

        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Predicted Happiness Score'])
        st.table(results_df)
        st.success("Predictions generated successfully!")
