import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib


def train_production_model():
    print("=" * 50)
    print("STARTING ECO PULSE ML PRODUCTION PIPELINE")
    print("=" * 50)

    # 1. Resolve absolute file paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "AEP_hourly.csv")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"CRITICAL ERROR: Cannot find dataset at {dataset_path}")
        return

    print(f"Loading real-world PJM grid dataset: {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # 2. Chronological Ordering and High-Leverage Feature Engineering
    print("Extracting time-series patterns from Datetime column...")
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

        # Pull out human behavioral, seasonal, and macro yearly trends
        df["Hour"] = df["Datetime"].dt.hour
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
        df["Month"] = df["Datetime"].dt.month
        df["DayOfYear"] = df["Datetime"].dt.dayofyear
        df["Year"] = df["Datetime"].dt.year  # Added macro tracker feature
    else:
        print("CRITICAL ERROR: 'Datetime' column missing from dataset.")
        return

    # 3. Feature Mapping to our 5 cyclical and macro drivers
    feature_cols = ["Hour", "DayOfWeek", "Month", "DayOfYear", "Year"]
    target_col = "AEP_MW"

    X = df[feature_cols].values
    y = df[target_col].values

    # 4. Generate Validation Splits (80% Train, 20% Test)
    print("Splitting dataset into Shuffled Training and Testing subsets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training Matrix Size: {len(X_train)} rows | Evaluation Split Size: {len(X_test)} rows")

    # 5. Feature Scaling Alignment
    print("Normalizing feature distributions via StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Fit Tuned XGBoost Regressor
    print("Fitting XGBoost Hyperparameter Regression Engine...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)

    # 7. Generate Evaluation Performance Metrics
    print("Evaluating trained model against test split...")
    y_pred = xgb_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 8. Render Production Metrics Report Card
    print("\n" + "=" * 48)
    print("       XGBOOST PRODUCTION PERFORMANCE REPORT    ")
    print("=" * 48)
    print(f" Mean Absolute Error (MAE):     {mae:.2f} MW")
    print(f" Root Mean Squared Error (RMSE): {rmse:.2f} MW")
    print(f" R-squared Score (R² Accuracy):  {r2 * 100:.2f}%")
    print("=" * 48 + "\n")

    # 9. Persist Binary Modeling Artifacts
    model_output_path = os.path.join(models_dir, "eco_pulse_xgb.joblib")
    scaler_output_path = os.path.join(models_dir, "scaler.joblib")
    metrics_output_path = os.path.join(models_dir, "model_metrics.json")

    joblib.dump(xgb_model, model_output_path)
    joblib.dump(scaler, scaler_output_path)

    # 10. Write Analytics Summary JSON
    metrics_payload = {
        "r2_score_accuracy_pct": round(r2 * 100, 2),
        "mean_absolute_error_mw": round(mae, 2),
        "root_mean_squared_error_mw": round(rmse, 2),
        "training_sample_count": len(X),
        "last_trained_timestamp": datetime.now().isoformat()
    }

    with open(metrics_output_path, "w") as json_file:
        json.dump(metrics_payload, json_file, indent=4)
    print(f"Saved performance audit dashboard logs: {metrics_output_path}")

    print("\n" + "=" * 50)
    print(" PIPELINE EXECUTION COMPLETE: MODEL RE-READY FOR API ")
    print("=" * 50)


if __name__ == "__main__":
    train_production_model()