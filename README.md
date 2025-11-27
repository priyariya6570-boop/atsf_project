Advanced Time Series Forecasting with Neural Networks and Explainability (XAI)
This project implements an end‑to‑end multivariate time series forecasting pipeline using a deep learning model (LSTM) and a statistical baseline model (Prophet), along with Explainable AI (XAI) analysis using SHAP. The goal is to build a multi‑step forecasting system, compare performance against a strong baseline, and interpret the model’s predictions.

Project Overview
Generate a synthetic multivariate time series dataset with trend, seasonality, and at least 5 correlated features.

Train an LSTM‑based sequence model for next‑step forecasting.

Train a Prophet model as a statistical baseline on the same target series.

Compare performance of LSTM vs Prophet using RMSE and MAE on a held‑out test set.

Apply SHAP to understand which features and time steps contribute most to LSTM predictions.

Provide well‑structured, reproducible Python code suitable for evaluation and reuse.

Project Structure
text
atsf_project/
├── data/
│   └── synthetic_multivariate.csv      # Generated multivariate time series
├── src/
│   ├── generate_data.py                # Synthetic data generation
│   ├── train_lstm.py                   # LSTM model training + evaluation
│   ├── baseline_prophet.py             # Prophet baseline model + evaluation
│   ├── explain_shap.py                 # SHAP explainability for LSTM
│   └── utils.py                        # Common helpers (paths, etc.)
├── lstm_model.h5                       # Trained LSTM model
├── X_test.npy, y_test_inv.npy          # Saved test data (for SHAP and analysis)
├── x_scaler_*.npy, y_scaler_*.npy      # Saved scalers for inverse transforms
├── shap_feature_importance.png         # SHAP feature importance plot
├── main.py                             # Orchestrates full pipeline
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files/folders excluded from git
└── README.md                           # Project documentation
Data Generation
src/generate_data.py programmatically creates a synthetic multivariate time series:

Number of time steps: 400 (configurable).

Number of features: 5 (feature_1 … feature_5), each with:

Linear trend.

Seasonal pattern (sinusoidal).

Gaussian noise.

Target variable:

Linear combination of selected features plus noise to ensure correlation.

The script writes data/synthetic_multivariate.csv, which is used by both models.

Run:

bash
python -m src.generate_data
Models
LSTM Deep Learning Model
src/train_lstm.py:

Preprocessing:

Scales features and target to using MinMaxScaler.​

Builds supervised sequences with a sliding window of 20 time steps and 1‑step‑ahead target.

Architecture:

LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dense(1).

Optimizer: Adam.

Loss: MSE.

Early stopping on validation loss.

Evaluation:

Inverse‑transforms predictions to original scale.

Computes RMSE and MAE on the test set.

Artifacts:

Saves lstm_model.h5 and scaler parameters.

Saves X_test.npy and y_test_inv.npy for later XAI.

Run:

bash
python -m src.train_lstm
The script prints LSTM RMSE and MAE and stores all artifacts.

Prophet Baseline Model
src/baseline_prophet.py:

Uses the same target series as the LSTM.

Wraps the series in Prophet’s required ds (date) and y columns.

Splits the series into train and test (80/20).

Fits a Prophet model on the training data.

Forecasts on the test horizon and computes RMSE and MAE.

Run:

bash
python -m src.baseline_prophet
This gives baseline metrics for direct comparison with the LSTM.

Explainable AI (SHAP)
src/explain_shap.py provides model interpretability:

Loads lstm_model.h5 and X_test.npy.

Flattens the 3‑D test sequences (time_steps × features) into 2‑D tabular features.

Uses SHAP KernelExplainer with a small background set and several test samples.

Computes SHAP values and aggregates mean absolute importance per feature.

Creates a horizontal bar plot of top features and saves it to shap_feature_importance.png.

Run:

bash
python -m src.explain_shap
Use this figure in the report to explain which lagged features most influence the LSTM’s forecasts.

Full Pipeline
To run all steps in order:

bash
python main.py
This will:

Generate the dataset.

Train and evaluate the LSTM.

Train and evaluate the Prophet baseline.

Run SHAP explainability and save the importance plot.

Environment Setup
Create and activate a virtual environment (example):

bash
python -m venv env
env\Scripts\activate        # Windows
Install dependencies:

bash
pip install --upgrade pip
pip install -r requirements.txt
The requirements.txt file includes the main libraries used:

numpy

pandas

matplotlib

scikit-learn

tensorflow

prophet

shap

Results Summary
LSTM achieves lower RMSE and MAE than the Prophet baseline on the synthetic multivariate series, demonstrating the benefit of a deep sequence model for this task.

The SHAP analysis highlights which features and lags contribute most to the forecasted target values, supporting model transparency and explainability.

How to Reproduce
Clone the repository:

bash
git clone https://github.com/priyariya6570-boop/atsf_project.git
cd atsf_project
Create and activate the virtual environment, then install dependencies.

Run:

bash
python main.py
Inspect:

Console output for LSTM and Prophet metrics.

shap_feature_importance.png for feature importance.

Generated CSV and model artifacts in the repo folders.