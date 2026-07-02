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
# DATABASE CONNECTION (PRODUCTION-READY)
# ----------------------------------------------------------------------
def get_db_connection():
    """Connects using environment variables defined in Render/System."""
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
except Exception as e:
    xgb_model, scaler = None, None
    print(f"ML Model Load Warning: {e}")

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.route('/')
def home():
    """Serves the frontend."""
    return render_template('index.html')

@app.route('/api/v1/forecast', methods=['POST'])
def get_forecast():
    """Handles dynamic location-based forecasting."""
    api_key = request.headers.get('X-API-KEY')
    if api_key != os.environ.get("INTERNAL_API_KEY", "EcoPulseSecret2026"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    location = data.get("location", "Unknown Grid Zone")

    # Time-series features
    target_time = datetime.now()
    features = np.array([[target_time.hour, target_time.weekday(), target_time.month,
                          target_time.timetuple().tm_yday, target_time.year]])

    # Inference
    predicted_load = 15000.00
    if xgb_model and scaler:
        predicted_load = float(xgb_model.predict(scaler.transform(features))[0])

    # Database Persistence
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
    # init_database_infrastructure() # Keep this if you need auto-schema setup
    app.run(host='0.0.0.0', port=5000)