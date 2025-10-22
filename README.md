World Happiness Score — Streamlit Demo

This repository contains a small demo for predicting World Happiness Scores using several regression models and a Streamlit UI.

Contents

app.py — Streamlit application that accepts feature values and shows model predictions.

2015.csv — CSV dataset used for exploratory work and training.

models.pkl, scaler.pkl — model and preprocessing artifacts expected by the app (not included in the repo).

task.ipynb — notebook with EDA and training steps used while developing the project.

requirements.txt — Python dependencies.

Quick start (Windows PowerShell)

Create a virtual environment and activate it (recommended):

python -m venv .venv
.\.venv\Scripts\Activate.ps1


Install dependencies:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


Ensure the following artifact files exist in the project root (the app will error if they are missing):

models.pkl — a pickled dict (or iterable) of trained scikit-learn regression models.

scaler.pkl — a pickled scaler used for feature scaling of X.

If you don't have these files, you can recreate them by running the training steps in task.ipynb and saving artifacts using pickle.

Run the Streamlit app:

streamlit run app.py

Using the app

The UI lets you enter nine features used by the models:
Happiness Rank, Standard Error, Economy (GDP per Capita), Family, Health (Life Expectancy), Freedom, Trust (Government Corruption), Generosity, Dystopia Residual.

Click "Predict Happiness Score" to generate predictions from the saved models.

Notes

Model outputs are presented directly, without any additional transformation.

Make sure models.pkl and scaler.pkl match the preprocessing and model training pipeline used in task.ipynb.

Quick debugging tips

If predictions seem incorrect, check that the input feature order and ranges match what the models were trained on.

You can also inspect task.ipynb to review training, preprocessing, and scaling steps.

Development notes

The app was updated to load artifacts safely and show helpful errors when artifacts are missing.

If retraining models, follow the steps in task.ipynb and save models.pkl and scaler.pkl together.