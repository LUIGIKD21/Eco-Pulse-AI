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
# ML ENGINE LOADING
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "eco_pulse_xgb.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
    encoder = joblib.load(os.path.join(BASE_DIR, "models", "location_encoder.joblib"))
    print("Success: New model, scaler, and encoder loaded.")
except Exception as err:
    xgb_model, scaler, encoder = None, None, None
    print(f"Warning: ML artifacts could not be loaded: {err}")


# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@app.route('/api/v1/forecast', methods=['POST'])
def get_forecast():
    data = request.get_json() or {}
    location = data.get("location", "PowerConsumption_Zone1")
    target_time = datetime.now()

    predicted_load = 0.0
    if xgb_model and scaler and encoder:
        try:
            # 1. Transform Location string to Zone_Code
            zone_code = encoder.transform([location])[0]

            # 2. Features: [Temp, Humidity, Wind, Hour, Weekday, Month, DayOfYear, Year, Zone_Code]
            # Using defaults for weather until a live API is integrated
            features = np.array([[22.0, 50.0, 5.0, target_time.hour,
                                  target_time.weekday(), target_time.month,
                                  target_time.timetuple().tm_yday, target_time.year,
                                  zone_code]])

            scaled_features = scaler.transform(features)
            predicted_load = float(xgb_model.predict(scaled_features)[0])
        except Exception as e:
            print(f"Inference error: {e}")

    return jsonify({
        "status": "success",
        "location": location,
        "forecast": {"predicted_load_kwh": round(predicted_load, 2)},
        "timestamp": target_time.isoformat()
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)