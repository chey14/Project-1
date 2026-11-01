import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

os.makedirs("results_dashboard", exist_ok=True)
np.random.seed(42)

# ---- same data and same split as 5_transformer_forecast.py ----
df = pd.read_csv("data/House_1_daily.csv", parse_dates=["Time"]).dropna().sort_values("Time").reset_index(drop=True)
vals = df["Aggregate"].values
times = df["Time"]
n = len(vals)
train_end, val_end = int(n * 0.70), int(n * 0.85)
L = 7

def make_xy(start, end, calendar=False):
    X, y = [], []
    for i in range(start, end):
        feats = list(vals[i - L : i])              # previous 7 daily means
        if calendar:
            feats += [times.iloc[i].month, times.iloc[i].dayofweek]
        X.append(feats); y.append(vals[i])
    return np.array(X), np.array(y)

def metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mape = float(np.nanmean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape}

def train_xgb(calendar):
    X_tr, y_tr = make_xy(L, train_end, calendar)
    X_va, y_va = make_xy(train_end, val_end, calendar)
    X_te, y_te = make_xy(val_end, n, calendar)
    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    secs = time.time() - t0
    return model, model.predict(X_te), y_te, X_te, secs

# 1) XGBoost, lags only (fair comparison with the Transformer)
m1, pred_xgb, actual, X_te, t1 = train_xgb(calendar=False)

# 2) XGBoost + month and day-of-week features
m2, pred_xgb_cal, _, _, t2 = train_xgb(calendar=True)

# 3) Persistence baseline: tomorrow = today
pred_persist = X_te[:, -1]

res = pd.DataFrame({
    "Time": times.iloc[val_end:n].reset_index(drop=True),
    "Actual": actual,
    "XGB": pred_xgb,
    "XGB_Calendar": pred_xgb_cal,
    "Persistence": pred_persist,
})
res.to_csv("results_dashboard/xgb_results.csv", index=False)

out = {
    "XGBoost": {**metrics(actual, pred_xgb), "best_iteration": int(m1.best_iteration), "train_seconds": round(t1, 2)},
    "XGBoost + Calendar": {**metrics(actual, pred_xgb_cal), "best_iteration": int(m2.best_iteration), "train_seconds": round(t2, 2)},
    "Persistence": metrics(actual, pred_persist),
}
with open("results_dashboard/xgb_metrics.json", "w") as f:
    json.dump(out, f, indent=2)

for k, v in out.items():
    print(f"{k:20s} MAE={v['MAE']:.2f}  RMSE={v['RMSE']:.2f}  MAPE={v['MAPE']:.2f}%")
print("[OK] XGBoost results saved.")