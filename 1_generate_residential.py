import os
import numpy as np
import pandas as pd

np.random.seed(42)
START_DATE = "2022-01-01 00:00:00"
NUM_DAYS   = 1095  # 3 years
FREQ       = "1min"
OUT_DIR  = "data"
os.makedirs(OUT_DIR, exist_ok=True)
RAW_FILE = os.path.join(OUT_DIR, "House_1_smartmeter_raw.csv")

periods    = NUM_DAYS * 24 * 60
time_index = pd.date_range(start=START_DATE, periods=periods, freq=FREQ)
hours   = np.asarray(time_index.hour,          dtype=np.float64)
minutes = np.asarray(time_index.minute,        dtype=np.float64)
dow     = np.asarray(time_index.day_of_week)   
month   = np.asarray(time_index.month,         dtype=np.float64)
dayfrac = hours + minutes / 60.0
day_of_year = np.asarray(time_index.day_of_year, dtype=np.float64)

# Appliance loads
fridge_cycle = (np.sin(2 * np.pi * dayfrac / 4.0) > 0.3).astype(float)
fridge_kw    = np.clip(0.12 + 0.05 * fridge_cycle + np.random.normal(0, 0.01, periods), 0.05, 0.30)
ac_seasonal = 0.5 + 0.5 * np.abs(np.cos((month - 1) / 12 * 2 * np.pi))
ac_evening  = np.exp(-((dayfrac - 20.0) ** 2) / (2 * 2.0 ** 2))
ac_kw       = np.clip(ac_seasonal * (0.3 + 1.2 * ac_evening) + np.random.normal(0, 0.05, periods), 0.0, 2.0)
wh_morning = np.exp(-((dayfrac - 7.0)  ** 2) / (2 * 0.8 ** 2))
wh_evening = np.exp(-((dayfrac - 20.0) ** 2) / (2 * 1.0 ** 2))
waterheater_kw = np.clip(1.2 * (wh_morning + wh_evening) + np.random.normal(0, 0.05, periods), 0.0, 2.5)
light_window = ((dayfrac >= 17.5) & (dayfrac <= 23.5)).astype(float)
lighting_kw  = np.clip(light_window * (0.15 + 0.15 * np.random.rand(periods)), 0.0, 0.40)
misc_kw = np.clip(0.08 + np.random.normal(0, 0.01, periods), 0.03, 0.15)

aggregate_kw = np.clip(fridge_kw + ac_kw + waterheater_kw + lighting_kw + misc_kw + np.random.normal(0, 0.03, periods), 0.15, None)

# Electrical
voltage_v = np.clip(230.0 - 4.0 * (aggregate_kw / aggregate_kw.max()) + np.random.normal(0, 1.5, periods), 215, 245)
power_factor = np.clip(0.98 - 0.05 * (aggregate_kw / aggregate_kw.max()) + np.random.normal(0, 0.005, periods), 0.85, 0.99)
current_a = np.clip((aggregate_kw * 1000.0) / (voltage_v * power_factor), 0.1, None)
frequency_hz = np.clip(50.0 + np.random.normal(0, 0.08, periods), 49.5, 50.5)
temp_seasonal = 25.0 + 8.0 * np.sin((day_of_year - 100) / 365 * 2 * np.pi)
temperature_c = temp_seasonal + 4.0 * np.sin((dayfrac - 9.0) / 24.0 * 2 * np.pi) + np.random.normal(0, 0.8, periods)

# Anomalies
n_spike, n_dip, n_outage = 450, 250, 60
spike_idx, dip_idx, outage_idx = np.random.choice(periods, n_spike, replace=False), np.random.choice(periods, n_dip, replace=False), np.random.choice(periods, n_outage, replace=False)
aggregate_kw[spike_idx] += np.random.uniform(2.0, 4.0, n_spike); ac_kw[spike_idx] += np.random.uniform(1.5, 3.0, n_spike)
aggregate_kw[dip_idx] = np.random.uniform(0.01, 0.05, n_dip); current_a[dip_idx] = np.random.uniform(0.05, 0.2, n_dip)
aggregate_kw[outage_idx] = 0.0
for _ in range(5):
    start = np.random.randint(0, periods - 120)
    lighting_kw[start:start + 120] = 0.30
aggregate_kw = np.clip(aggregate_kw, 0, None)

# Flags
quality_flag = np.array(["GOOD"] * periods, dtype=object); meter_status = np.array(["ACTIVE"] * periods, dtype=object)
n_missing = int(periods * 0.005); missing_idx = np.random.choice(periods, size=n_missing, replace=False)
aggregate_kw[missing_idx] = np.nan; quality_flag[missing_idx] = "MISSING"; meter_status[missing_idx] = "COMM_LOSS"

df = pd.DataFrame({
    "Meter_ID": "MH01SM0004821", "Consumer_ID": "CNS-2019-0083742", "Connection_Type": "Domestic", "Tariff_Category": "LT-I Residential",
    "Timestamp": time_index, "Aggregate_kW": np.round(aggregate_kw, 3), "Fridge_kW": np.round(fridge_kw, 2),
    "AC_kW": np.round(ac_kw, 2), "WaterHeater_kW": np.round(waterheater_kw, 2), "Lighting_kW": np.round(lighting_kw, 2),
    "Misc_kW": np.round(misc_kw, 2), "Voltage_V": np.round(voltage_v, 1), "Current_A": np.round(current_a, 2),
    "PowerFactor": np.round(power_factor, 3), "Frequency_Hz": np.round(frequency_hz, 2), "Temperature_C": np.round(temperature_c, 1),
    "Meter_Status": meter_status, "Quality_Flag": quality_flag
})
df.to_csv(RAW_FILE, index=False)
print(f"[OK] Residential Data Generated: {RAW_FILE}")