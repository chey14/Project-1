import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

os.makedirs("results_dashboard", exist_ok=True)
df = pd.read_csv("data/House_1_1min.csv", parse_dates=["Time"])
df = df.sort_values("Time").reset_index(drop=True).dropna(subset=["Aggregate"])
df = df.iloc[:150000].copy()

HALF = 10
power = df["Aggregate"].values
gradient = np.diff(power, prepend=power[0])

X, times = [], []
for i in range(HALF, len(power) - HALF):
    power_window = power[i - HALF : i + HALF + 1]
    grad_window  = gradient[i - HALF : i + HALF + 1]
    X.append(np.concatenate([power_window, grad_window]))
    times.append(df["Time"].iloc[i])

X = np.array(X)
model = IsolationForest(n_estimators=150, contamination=0.0015, random_state=42, n_jobs=-1)
model.fit(X)

preds = model.predict(X)
scores = -model.score_samples(X)

results = pd.DataFrame({
    "Time": times, "Center_Power": power[HALF:-HALF],
    "Anomaly": (preds == -1).astype(int), "Anomaly_Score": scores,
})
results.to_csv("results_dashboard/anomaly_results.csv", index=False)
print(f"[OK] Anomalies found: {results['Anomaly'].sum()}")