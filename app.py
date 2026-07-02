import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import psycopg2
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)


# ----------------------------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------------------------
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port="5432"
    )


# ----------------------------------------------------------------------
# ML ENGINE LOADING (WITH DEBUGGING)
# ----------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "eco_pulse_xgb.joblib")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.joblib")

    xgb_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("Success: Real-world trained XGBoost grid model and scaler loaded.")
except Exception as err:
    xgb_model = None
    scaler = None
    print(f"Warning: ML model artifacts could not be loaded ({err}).")

# DEBUGGING PRINTS
print(f"DEBUG: Model loaded? {xgb_model is not None}")
print(f"DEBUG: Scaler loaded? {scaler is not None}")


# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/v1/forecast', methods=['POST'])
def get_forecast():
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.environ.get('INTERNAL_API_KEY', 'EcoPulseSecret2026')

    if api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    location = data.get("location", "Unknown Grid Zone")
    target_time = datetime.now()

    # 1. ML Inference Engine Calculation
    predicted_load = 0.0
    if xgb_model and scaler:
        try:
            input_features = np.array([[target_time.hour, target_time.weekday(),
                                        target_time.month, target_time.timetuple().tm_yday,
                                        target_time.year]])
            scaled_features = scaler.transform(input_features)
            predicted_load = float(xgb_model.predict(scaled_features)[0])
        except Exception as ml_err:
            print(f"Inference error: {ml_err}")

    # 2. Database Persistence
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO prediction_logs
               (timestamp, location, actual_temp, humidity, predicted_kwh, confidence_interval)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (target_time, location, 22.2, 55.0, predicted_load, 0.95)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as db_err:
        print(f"Database logging skipped: {db_err}")

    return jsonify({
        "status": "success",
        "location": location,
        "forecast": {"predicted_load_mw": round(predicted_load, 2)},
        "timestamp": target_time.isoformat()
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)