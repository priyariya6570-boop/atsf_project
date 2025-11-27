from src.generate_data import main as gen_main
from src.train_lstm import main as lstm_main
from src.baseline_prophet import main as prophet_main
from src.explain_shap import main as shap_main

if __name__ == "__main__":
    print("Step 1: Generating data...")
    gen_main()
    print("\nStep 2: Training LSTM model...")
    lstm_main()
    print("\nStep 3: Training Prophet baseline...")
    prophet_main()
    print("\nStep 4: Running SHAP explainability...")
    shap_main()
    print("\nPipeline finished.")
