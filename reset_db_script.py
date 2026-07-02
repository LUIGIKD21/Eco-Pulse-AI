import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def nuke_database():
    try:
        # Connect to the default 'postgres' system database
        conn = psycopg2.connect(
            host="localhost",
            user="postgres",
            password="admin123"  # Put your actual DB password here!
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Drop the eco_pulse_db
        print("Dropping eco_pulse_db...")
        cur.execute('DROP DATABASE IF EXISTS eco_pulse_db;')
        print("Database dropped successfully.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    nuke_database()

    ### Administrative: Reset Database Table
    POST
    http: // 127.0
    .0
    .1: 5000 / api / v1 / admin / reset_db
    Content - Type: application / json
    X - API - KEY: EcoPulseSecret2026