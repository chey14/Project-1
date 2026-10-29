import os, json
import numpy as np
import pandas as pd

os.makedirs("results_dashboard", exist_ok=True)
raw = pd.read_csv("data/House_1_smartmeter_raw.csv", parse_dates=["Timestamp"])
raw = raw[raw["Quality_Flag"] == "GOOD"].copy().sort_values("Timestamp")

hourly = raw.set_index("Timestamp")["Aggregate_kW"].resample("h").mean()
hour_profile = hourly.groupby(hourly.index.hour).mean().round(3)
day_profile = hourly.groupby(hourly.index.day_of_week).mean().round(3)
dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
day_profile.index = [dow_map[i] for i in day_profile.index]

appliance_cols = ["Fridge_kW", "AC_kW", "WaterHeater_kW", "Lighting_kW", "Misc_kW"]
appliance_totals_kwh = (raw[appliance_cols].sum() / 60.0).round(2)
total_kwh = float(appliance_totals_kwh.sum())
appliance_share_pct = (appliance_totals_kwh / total_kwh * 100).round(2)
raw["hour"] = raw["Timestamp"].dt.hour
dominant_appliance = raw.groupby("hour")[appliance_cols].mean().round(3).idxmax(axis=1).to_dict()

v_score = np.clip(1.0 - np.abs(raw["Voltage_V"].values - 230) / 15.0, 0, 1).mean()
p_score = np.clip((raw["PowerFactor"].values - 0.85) / 0.14, 0, 1).mean()
f_score = np.clip(1.0 - np.abs(raw["Frequency_Hz"].values - 50.0) / 0.5, 0, 1).mean()
pqi = round((0.5 * v_score + 0.3 * p_score + 0.2 * f_score) * 100, 1)

health = {}
for col in ["Fridge_kW", "AC_kW", "WaterHeater_kW"]:
    series = raw.set_index("Timestamp")[col]
    z = (series.tail(24 * 60).mean() - series.mean()) / (series.std() + 1e-9)
    health[col] = {"mean_kW": round(float(series.tail(24*60).mean()), 3), "baseline_kW": round(float(series.mean()), 3), "z_score": round(float(z), 2), "status": "ALERT" if abs(z) > 2 else "NORMAL"}

def estimate_bill(kwh):
    slabs = [(100, 3.50), (300, 4.80), (500, 6.20), (float("inf"), 7.50)]
    bill, prev = 0.0, 0.0
    for limit, rate in slabs:
        if kwh <= prev: break
        bill += (min(kwh, limit) - prev) * rate; prev = limit
    return round(bill, 2)

monthly_kwh = raw.set_index("Timestamp")["Aggregate_kW"].resample("ME").sum() / 60.0
monthly_bill = monthly_kwh.apply(estimate_bill)

out = {
    "hour_profile": {int(k): float(v) for k, v in hour_profile.items()},
    "day_profile": {k: float(v) for k, v in day_profile.items()},
    "peak_hour": int(hour_profile.idxmax()), "peak_day": day_profile.idxmax(),
    "appliance_totals_kwh": {k: float(v) for k, v in appliance_totals_kwh.items()},
    "appliance_share_pct": {k: float(v) for k, v in appliance_share_pct.items()},
    "dominant_appliance_per_hour": {int(k): str(v) for k, v in dominant_appliance.items()},
    "total_kwh": total_kwh, "power_quality_index": pqi,
    "pqi_breakdown": {"voltage_score": round(float(v_score)*100, 1), "powerfactor_score": round(float(p_score)*100, 1), "frequency_score": round(float(f_score)*100, 1)},
    "appliance_health": health,
    "monthly_kwh": {str(k.date()): float(v) for k, v in monthly_kwh.items()},
    "monthly_bill": {str(k.date()): float(v) for k, v in monthly_bill.items()},
}
with open("results_dashboard/utility_results.json", "w") as f: json.dump(out, f, indent=2)
print("[OK] Bonus analytics saved.")