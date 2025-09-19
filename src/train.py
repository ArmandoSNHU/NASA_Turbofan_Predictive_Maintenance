# src/train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from preprocess import load_fd001, add_rul
import joblib

if __name__ == "__main__":
    df = load_fd001()     # uses full or demo automatically
    df = add_rul(df)

    print("Dataset shape:", df.shape)
    print(df.head())

    feature_cols = [c for c in df.columns if c.startswith(("sensor_", "op_setting_"))]
    X, y = df[feature_cols], df["RUL"]

    test_size = 0.2 if len(df) >= 10 else 0.5
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    print(f"✅ Validation RMSE: {rmse:.2f}")

    # Save model
    Path("reports").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "reports/rf_fd001.joblib")
    print("💾 Saved model -> reports/rf_fd001.joblib")
