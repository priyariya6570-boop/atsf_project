import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from .utils import get_data_path

def main() -> None:
    df = pd.read_csv(get_data_path("synthetic_multivariate.csv"))
    n = len(df)
    df_prophet = pd.DataFrame()
    df_prophet["ds"] = pd.date_range(start="2020-01-01", periods=n, freq="D")
    df_prophet["y"] = df["target"].values

    split = int(0.8 * n)
    train_df = df_prophet.iloc[:split]
    test_df = df_prophet.iloc[split:]

    model = Prophet()
    model.fit(train_df)

    forecast = model.predict(test_df[["ds"]])
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Prophet RMSE: {rmse:.4f}")
    print(f"Prophet MAE : {mae:.4f}")

if __name__ == "__main__":
    main()
