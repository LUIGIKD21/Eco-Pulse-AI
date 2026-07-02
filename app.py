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
# DATABASE INFRASTRUCTURE
# ----------------------------------------------------------------------
def init_database_infrastructure():
    """Initializes the prediction_logs table if it does not exist."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            database=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            port="5432"
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                location VARCHAR(100) NOT NULL,
                actual_temp NUMERIC(5, 2) NOT NULL,
                humidity NUMERIC(5, 2) NOT NULL,
                predicted_kwh NUMERIC(10, 2) NOT NULL,
                confidence_interval NUMERIC(3, 2) NOT NULL
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Database schema verified/created successfully.")
    except Exception as err:
        print(f"Database initialization error: {err}")

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port="5432"
    )

# ----------------------------------------------------------------------
# ML ENGINE LOADING
# ----------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "eco_pulse_xgb.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
    print("Success: XGBoost model and scaler loaded.")
except Exception as err:
    xgb_model, scaler = None, None
    print(f"Warning: ML model artifacts could not be loaded ({err}).")

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

    predicted_load = 0.0
    if xgb_model and scaler:
        try:
            features = np.array([[target_time.hour, target_time.weekday(),
                                  target_time.month, target_time.timetuple().tm_yday,
                                  target_time.year]])
            predicted_load = float(xgb_model.predict(scaler.transform(features))[0])
        except Exception as e:
            print(f"Inference error: {e}")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prediction_logs (timestamp, location, actual_temp, humidity, predicted_kwh, confidence_interval) VALUES (%s, %s, %s, %s, %s, %s)",
            (target_time, location, 22.2, 55.0, predicted_load, 0.95)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

    return jsonify({
        "status": "success",
        "location": location,
        "forecast": {"predicted_load_mw": round(predicted_load, 2)},
        "timestamp": target_time.isoformat()
    }), 200

if __name__ == '__main__':
    init_database_infrastructure()
    app.run(host='0.0.0.0', port=5000)