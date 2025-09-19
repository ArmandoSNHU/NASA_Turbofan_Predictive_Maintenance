# src/inspect_model.py
# ---------------------------------------------
# Load the trained RandomForest model and inspect it
# - prints parameters
# - shows number of trees
# - saves feature importance chart to reports/figs/
# ---------------------------------------------
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path("reports/rf_fd001.joblib")
OUT_FIG = Path("reports/figs/feature_importances.png")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Missing model: {MODEL_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    print("✅ Loaded model")
    print(model)
    print("\nType:", type(model))
    print("\nParameters:", model.get_params())
    print("Number of trees:", len(model.estimators_))

    # Feature importances
    importances = pd.Series(model.feature_importances_)
    importances = importances.sort_values(ascending=False)

    # Plot top 15
    Path("reports/figs").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    importances.head(15).plot(kind="bar")
    plt.title("Top 15 Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"📊 Saved feature importance chart -> {OUT_FIG.resolve()}")

if __name__ == "__main__":
    main()
