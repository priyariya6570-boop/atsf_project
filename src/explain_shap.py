import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt


def main() -> None:
    # Load trained model and test data
    model = tf.keras.models.load_model("lstm_model.h5")
    X_test = np.load("X_test.npy")   # shape: (N, time_steps, n_features)

    N, time_steps, n_features = X_test.shape

    # Flatten time dimension -> (N, time_steps * n_features)
    X_flat = X_test.reshape(N, time_steps * n_features)

    # Background and test samples
    background = X_flat[:50, :]
    test_samples = X_flat[50:60, :]   # (10, F)

    # Prediction function: flat -> 3D -> model
    def predict_fn(x):
        x = np.array(x)
        x_3d = x.reshape(x.shape[0], time_steps, n_features)
        return model.predict(x_3d, verbose=0)

    explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(test_samples, nsamples=100)

    # KernelExplainer returns a list; take the first element and ensure 2D
    shap_vals = np.array(shap_values[0])  # shape (10, F)

    # Compute mean absolute SHAP value for each feature (global importance)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)  # shape (F,)

    # Plot top 20 features for simplicity
    top_k = 20
    idx = np.argsort(mean_abs)[-top_k:]
    top_importance = mean_abs[idx]
    feature_idx = np.arange(len(idx))

    plt.figure(figsize=(8, 4))
    plt.barh(feature_idx, top_importance)
    plt.yticks(feature_idx, [f"f_{i}" for i in idx])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top feature importances for LSTM (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png")
    print("Saved SHAP plot to shap_feature_importance.png")


if __name__ == "__main__":
    main()
