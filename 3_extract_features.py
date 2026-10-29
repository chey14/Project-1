import os
import pandas as pd

os.makedirs("data", exist_ok=True)
raw = pd.read_csv("data/House_1_smartmeter_raw.csv", parse_dates=["Timestamp"])
clean = raw[raw["Quality_Flag"] == "GOOD"].copy()

minute_df = pd.DataFrame({"Time": clean["Timestamp"], "Aggregate": clean["Aggregate_kW"] * 1000.0})
minute_df.to_csv("data/House_1_1min.csv", index=False)

daily_df = minute_df.set_index("Time").resample("D")["Aggregate"].mean().reset_index()
daily_df.columns = ["Time", "Aggregate"]
daily_df = daily_df.dropna().reset_index(drop=True)
daily_df.to_csv("data/House_1_daily.csv", index=False)
print("[OK] Features extracted: House_1_1min.csv and House_1_daily.csv")