import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor

# 1. Load and Preprocess Data
df = pd.read_csv('powerconsumption.csv')
df_melted = df.melt(
    id_vars=['Datetime', 'Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows'],
    value_vars=['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'],
    var_name='Zone',
    value_name='PowerConsumption'
)

df_melted['Datetime'] = pd.to_datetime(df_melted['Datetime'])
df_melted['Hour'] = df_melted['Datetime'].dt.hour
df_melted['Weekday'] = df_melted['Datetime'].dt.weekday
df_melted['Month'] = df_melted['Datetime'].dt.month
df_melted['DayOfYear'] = df_melted['Datetime'].dt.dayofyear
df_melted['Year'] = df_melted['Datetime'].dt.year

# Encode Zones
encoder = LabelEncoder()
df_melted['Zone_Code'] = encoder.fit_transform(df_melted['Zone'])

# 2. Define Features and Target
# Using Environmental + Time + Location features
features = ['Temperature', 'Humidity', 'WindSpeed', 'Hour', 'Weekday', 'Month', 'DayOfYear', 'Year', 'Zone_Code']
X = df_melted[features]
y = df_melted['PowerConsumption']

# 3. Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_scaled, y)

# 4. Export Artifacts
joblib.dump(model, 'models/eco_pulse_xgb.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(encoder, 'models/location_encoder.joblib')

print("Training complete. Artifacts saved to models/ folder.")