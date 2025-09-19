# src/predict.py
# ------------------------------------------------------------
# Predict RUL for the test set's last cycle per engine (FD001)
# Uses:
#   - data/test_FD001.txt         (whitespace-delimited)
#   - data/RUL_FD001.txt          (one RUL per test engine, order 1..N)
#   - reports/rf_fd001.joblib     (saved by train.py)
# Outputs:
#   - reports/predictions_fd001.csv
# ------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from preprocess import load_fd001  # same loader you used in train.py

MODEL_PATH = Path("reports/rf_fd001.joblib")
TEST_PATH = Path("data/test_FD001.txt")
RUL_PATH = Path("data/RUL_FD001.txt")
OUT_CSV = Path("reports/predictions_fd001.csv")

def load_true_rul(rul_path: Path) -> pd.Series:
    """RUL file: one integer per test engine in order (1..N)."""
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing {rul_path.resolve()}")
    vals = []
    with rul_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(float(line)))  # tolerate floats like '112.0'
    return pd.Series(vals, name="true_RUL")

def main():
    # sanity checks
    for p in [MODEL_PATH, TEST_PATH, RUL_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p.resolve()}")

    # load artifacts
    model = joblib.load(MODEL_PATH)
    test_df = load_fd001(str(TEST_PATH))

    if len(test_df) == 0:
        raise ValueError("Test set parsed to zero rows. Check file formatting (whitespace-delimited).")

    # take last cycle per engine as "current health snapshot"
    last_by_unit = (
        test_df.sort_values(["unit", "cycle"])
               .groupby("unit", as_index=False)
               .tail(1)
               .sort_values("unit")
               .reset_index(drop=True)
    )

    feature_cols = [c for c in last_by_unit.columns if c.startswith(("sensor_", "op_setting_"))]
    X = last_by_unit[feature_cols]

    pred_rul = model.predict(X)
    pred_df = pd.DataFrame({
        "unit": last_by_unit["unit"].astype(int).values,
        "pred_RUL": pred_rul
    }).sort_values("unit").reset_index(drop=True)

    # true RUL order is 1..N per FD001 convention (matches our simulated generator too)
    true_rul = load_true_rul(RUL_PATH)
    if len(true_rul) != len(pred_df):
        raise ValueError(f"RUL length mismatch: true={len(true_rul)} vs preds={len(pred_df)}")

    pred_df["true_RUL"] = true_rul
    pred_df["error"] = pred_df["pred_RUL"] - pred_df["true_RUL"]

    # metrics
    rmse = float(np.sqrt(np.mean((pred_df["error"])**2)))
    mae  = float(np.mean(np.abs(pred_df["error"])))
    mape = float(np.mean(np.abs(pred_df["error"] / np.maximum(pred_df["true_RUL"], 1)))) * 100.0  # avoid div0

    # print a neat summary
    print("\nPredictions (first 10 engines):")
    print(pred_df.head(10).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nMetrics on test last-cycle per engine:")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE : {mae:,.2f}")
    print(f"  MAPE: {mape:,.1f}%")

    # save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved predictions -> {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
