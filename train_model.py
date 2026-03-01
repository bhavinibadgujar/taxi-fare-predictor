"""
Taxi Fare Prediction - ML Model Training Script
Generates synthetic dataset + trains Random Forest & XGBoost models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
N = 10000

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# Mumbai coordinates range
lat1 = np.random.uniform(18.90, 19.20, N)
lon1 = np.random.uniform(72.77, 73.05, N)
lat2 = np.random.uniform(18.90, 19.20, N)
lon2 = np.random.uniform(72.77, 73.05, N)
distance = haversine(lat1, lon1, lat2, lon2)
distance = np.clip(distance, 0.5, 40)

hour = np.random.randint(0, 24, N)
day_of_week = np.random.randint(0, 7, N)
passengers = np.random.randint(1, 7, N)
weather = np.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1])  # Clear, Rain, Storm

# Surge logic
surge = np.ones(N)
surge += np.where((hour >= 8) & (hour <= 10), 0.5, 0)
surge += np.where((hour >= 17) & (hour <= 20), 0.6, 0)
surge += np.where((day_of_week >= 5) & (hour >= 20), 0.4, 0)
surge += np.where(weather == 1, 0.3, 0)
surge += np.where(weather == 2, 0.7, 0)
surge = np.clip(surge, 1.0, 3.0)

is_peak = ((hour >= 8) & (hour <= 10) | (hour >= 17) & (hour <= 20)).astype(int)

# Base fare formula: base + per_km + surge + noise
base_fare = 30
per_km_rate = 14
fare = (base_fare + per_km_rate * distance * surge +
        passengers * 2 +
        np.random.normal(0, 8, N))
fare = np.clip(fare, 40, 2000)

df = pd.DataFrame({
    'distance_km': distance,
    'hour': hour,
    'day_of_week': day_of_week,
    'passengers': passengers,
    'weather': weather,
    'is_peak': is_peak,
    'surge_multiplier': surge,
    'fare': fare.round(2)
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
df.to_csv(os.path.join(BASE_DIR, 'data', 'taxi_fare_data.csv'), index=False)
print(f"✅ Dataset created: {len(df)} rows")
print(df.describe())

# ─────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────
X = df.drop('fare', axis=1)
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"   R² Score : {rf_r2:.4f}")
print(f"   MAE      : ₹{rf_mae:.2f}")

# XGBoost (using GradientBoosting as fallback)
print("\n⚡ Training XGBoost / GradientBoosting...")
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                       random_state=42, verbosity=0)
    model_name = "XGBoost"
except ImportError:
    xgb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, random_state=42)
    model_name = "GradientBoosting"

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"   R² Score : {xgb_r2:.4f}")
print(f"   MAE      : ₹{xgb_mae:.2f}")

# Save best model
best_model = xgb if xgb_r2 >= rf_r2 else rf
best_name = model_name if xgb_r2 >= rf_r2 else "RandomForest"
print(f"\n🏆 Best Model: {best_name} (R²={max(rf_r2, xgb_r2):.4f})")

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
model_save_path = os.path.join(BASE_DIR, 'models', 'fare_model.pkl')
with open(model_save_path, 'wb') as f:
    pickle.dump({'model': best_model, 'features': list(X.columns),
                 'model_name': best_name, 'r2': max(rf_r2, xgb_r2),
                 'mae': min(rf_mae, xgb_mae)}, f)

print(f"✅ Model saved to {model_save_path}")
