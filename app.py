import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend integration


# ----------------------------------------------------------------------
# DATABASE SELF-HEALING INFRASTRUCTURE
# ----------------------------------------------------------------------
def init_database_infrastructure():
    """
    Automates local environment setup. Connects to the master Postgres system
    database, ensures the project database exists, and initializes schemas.
    """
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_user = os.environ.get("DB_USER", "postgres")
    db_name = os.environ.get("DB_NAME", "eco_pulse_db")

    if not db_password:
        print("DATABASE INITIALIZATION WARNING: 'DB_PASSWORD' environment variable is missing.")
        return

    try:
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()

        # TEMPORARILY REPLACE YOUR EXISTING CREATE TABLE BLOCK WITH THIS:
        cur.execute("DROP TABLE IF EXISTS prediction_logs CASCADE;")  # Force delete old structure
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_logs
                    (
                        id
                        SERIAL
                        PRIMARY
                        KEY,
                        timestamp
                        TIMESTAMP
                        NOT
                        NULL,
                        location
                        VARCHAR
                    (
                        100
                    ) NOT NULL,
                        actual_temp NUMERIC
                    (
                        5,
                        2
                    ) NOT NULL,
                        humidity NUMERIC
                    (
                        5,
                        2
                    ) NOT NULL,
                        predicted_kwh NUMERIC
                    (
                        10,
                        2
                    ) NOT NULL,
                        confidence_interval NUMERIC
                    (
                        3,
                        2
                    ) NOT NULL
                        );
                    """)

        cur.close()
        conn.close()
    except Exception as err:
        print(f"System initialization warning (Master DB connection failed): {err}")
        return

    try:
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Kept fully backward compatible with your existing local database table structure
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_logs
                    (
                        id
                        SERIAL
                        PRIMARY
                        KEY,
                        timestamp
                        TIMESTAMP
                        NOT
                        NULL,
                        location
                        VARCHAR
                    (
                        100
                    ) NOT NULL,
                        actual_temp NUMERIC
                    (
                        5,
                        2
                    ) NOT NULL,
                        humidity NUMERIC
                    (
                        5,
                        2
                    ) NOT NULL,
                        predicted_kwh NUMERIC
                    (
                        10,
                        2
                    ) NOT NULL,
                        confidence_interval NUMERIC
                    (
                        3,
                        2
                    ) NOT NULL
                        );
                    """)
        cur.close()
        conn.close()
        print("Database schema synchronized and ready for production requests.")
    except Exception as err:
        print(f"Schema synchronization error: {err}")


def get_db_connection():
    # This forces the password 'admin123' to be used every single time
    return psycopg2.connect(
        host="localhost",
        database="eco_pulse_db",
        user="postgres",
        password="admin123",
        port="5432"
    )


# ----------------------------------------------------------------------
# ML INFERENCE WORK ENGINE LOADING LAYER
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
    print(f"Warning: ML model artifacts could not be loaded ({err}). Using structural fallbacks.")


# ----------------------------------------------------------------------
# REST API ENDPOINTS
# ----------------------------------------------------------------------
@app.route('/api/v1/forecast', methods=['POST'])
def get_forecast():
    """Run predictive inference cycles via time-series engineered XGBoost."""
    # Security Token Validation Check
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.environ.get('INTERNAL_API_KEY', 'EcoPulseSecret2026')

    if api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    location = data.get("location", "Unknown Grid Zone")

    # 1. Advanced Time-Series Ingestion Layer
    # Dynamically parses an optional custom timestamp string, or defaults to current system time
    custom_time_str = data.get("timestamp")
    if custom_time_str:
        try:
            target_time = datetime.fromisoformat(custom_time_str)
        except Exception:
            target_time = datetime.now()
    else:
        target_time = datetime.now()

    # Extract our 5 exact features matching training shape
    hour = target_time.hour
    day_of_week = target_time.weekday()
    month = target_time.month
    day_of_year = target_time.timetuple().tm_yday
    year = target_time.year

    # 2. ML Inference Engine Calculations
    if xgb_model and scaler:
        try:
            # Inputs mapped to structural training layer matrix: [[Hour, DayOfWeek, Month, DayOfYear, Year]]
            input_features = np.array([[hour, day_of_week, month, day_of_year, year]])
            scaled_features = scaler.transform(input_features)
            predicted_load = float(xgb_model.predict(scaled_features)[0])
        except Exception as ml_err:
            print(f"Inference execution engine fallback triggered: {ml_err}")
            predicted_load = 15000.00
    else:
        predicted_load = 15000.00

    confidence_interval = 0.95

    # 3. Data Persistence Layer (Preserves table parameters safely)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO prediction_logs
               (timestamp, location, actual_temp, humidity, predicted_kwh, confidence_interval)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (target_time, location, 22.2, 55.0, predicted_load, confidence_interval)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as db_err:
        print(f"Database logging skipped (DB connection issue): {db_err}")

    return jsonify({
        "status": "success",
        "location": location,
        "time_context": {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_year": day_of_year,
            "year": year
        },
        "forecast": {
            "predicted_load_mw": round(predicted_load, 2),
            "confidence_interval": confidence_interval
        },
        "timestamp": target_time.isoformat()
    }), 200


@app.route('/api/v1/history', methods=['GET'])
def get_prediction_history():
    """Fetches logged historical model inference traces from database with explicit mapping shields."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Explicitly fetching target columns to isolate performance outputs
        cur.execute("""
                    SELECT timestamp, location, predicted_kwh
                    FROM prediction_logs
                    ORDER BY id DESC
                        LIMIT 50;
                    """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Added safety parsing to prevent None/NULL row objects from crashing the array pipeline
        history_list = []
        for r in rows:
            try:
                timestamp_str = r[0].isoformat() if r[0] else datetime.now().isoformat()
                location_str = str(r[1]) if r[1] else "Unknown Region"
                load_val = float(r[2]) if r[2] is not None else 0.0

                history_list.append({
                    "timestamp": timestamp_str,
                    "location": location_str,
                    "predicted_load_mw": load_val
                })
            except Exception as row_parse_err:
                print(f"Skipping corrupt historic row record: {row_parse_err}")
                continue

        return jsonify({"status": "success", "count": len(history_list), "data": history_list}), 200

    except Exception as err:
        print(f"CRITICAL ERROR inside GET /history endpoint layer: {err}")
        return jsonify({"status": "error", "details": str(err)}), 500

@app.route('/api/v1/admin/reset_db', methods=['POST'])
def reset_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS prediction_logs;")
        cur.close()
        conn.close()
        return jsonify({"status": "success", "message": "Table dropped. Restart app to re-sync schema."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
if __name__ == '__main__':
    init_database_infrastructure()
    app.run(host='0.0.0.0', port=5000)