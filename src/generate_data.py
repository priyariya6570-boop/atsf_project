import numpy as np
import pandas as pd
from pathlib import Path

def create_multivariate_timeseries(
    n_steps: int = 400,
    n_features: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    t = np.arange(n_steps)

    data = {}
    for i in range(n_features):
        trend = 0.01 * (i + 1) * t
        seasonal = np.sin(2 * np.pi * t / 50 + i)
        noise = np.random.normal(0, 0.3, n_steps)
        data[f"feature_{i+1}"] = trend + seasonal + noise

    target_noise = np.random.normal(0, 0.2, n_steps)
    data["target"] = (
        0.5 * data["feature_1"]
        + 0.3 * data["feature_3"]
        + 0.2 * data["feature_5"]
        + target_noise
    )

    return pd.DataFrame(data)

def main() -> None:
    df = create_multivariate_timeseries()
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_multivariate.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")

if __name__ == "__main__":
    main()
