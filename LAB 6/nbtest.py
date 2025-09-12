import fastf1
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Ensure cache folder exists
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# Load a session (example: 2023 Bahrain GP Race)
session = fastf1.get_session(2023, 1, "R")
session.load()

# Extract laps and weather
laps = session.laps
weather = session.weather_data

# Sort before asof merge
laps = laps.sort_values("Time").reset_index(drop=True)
weather = weather.sort_values("Time").reset_index(drop=True)

# Merge nearest weather record (within 1 minute tolerance)
laps = pd.merge_asof(laps, weather, on="Time", direction="nearest", tolerance=pd.Timedelta("60s"))

# Select relevant features
data = laps[["Driver", "LapNumber", "LapTime", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Pressure"]].copy()

# Drop rows missing essential values
data.dropna(subset=["LapTime", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Pressure"], inplace=True)

# Convert LapTime to seconds
data["LapTime_s"] = data["LapTime"].dt.total_seconds()

# Create target: Fast if lap < median, else Slow
median_time = data["LapTime_s"].median()
data["LapClass"] = np.where(data["LapTime_s"] < median_time, "Fast", "Slow")

# Encode Driver to numeric
encoder = LabelEncoder()
data["DriverCode"] = encoder.fit_transform(data["Driver"])

# Features and target
X = data[["DriverCode", "LapNumber", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Pressure"]]
y = data["LapClass"]

print(f"✅ Dataset ready with {len(X)} samples, {X.shape[1]} features")

# Train/test split
if len(X) < 20:
    raise ValueError("Not enough data for training. Try another session.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Show some probability predictions
proba_df = pd.DataFrame(y_proba, columns=model.classes_)
proba_df["TrueClass"] = y_test.values
print("\n🎯 Sample Probability Predictions:\n", proba_df.head(10))

# Conclusion
fast_prob = np.mean(proba_df["Fast"])
slow_prob = np.mean(proba_df["Slow"])
print(f"\n✅ Conclusion: On average, probability of a lap being FAST = {fast_prob:.2f}, SLOW = {slow_prob:.2f}")
