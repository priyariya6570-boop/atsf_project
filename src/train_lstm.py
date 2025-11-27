import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from .utils import get_data_path

TIME_STEPS = 20          # window length
FORECAST_HORIZON = 1     # predict next step

def create_sequences(X, y, time_steps=TIME_STEPS, horizon=FORECAST_HORIZON):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - horizon + 1):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps + horizon - 1])
    return np.array(Xs), np.array(ys)

def main() -> None:
    csv_path = get_data_path("synthetic_multivariate.csv")
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c != "target"]
    X_raw = df[feature_cols].values
    y_raw = df["target"].values.reshape(-1, 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    y_pred_scaled = model.predict(X_test)
    y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = y_scaler.inverse_transform(y_pred_scaled)

    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"LSTM RMSE: {rmse:.4f}")
    print(f"LSTM MAE : {mae:.4f}")

    out_dir = Path(__file__).resolve().parents[1]
    model.save(out_dir / "lstm_model.h5")
    np.save(out_dir / "x_scaler_minmax.npy", x_scaler.data_min_)
    np.save(out_dir / "x_scaler_max.npy", x_scaler.data_max_)
    np.save(out_dir / "y_scaler_min.npy", y_scaler.data_min_)
    np.save(out_dir / "y_scaler_max.npy", y_scaler.data_max_)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test_inv.npy", y_test_inv)

if __name__ == "__main__":
    main()
