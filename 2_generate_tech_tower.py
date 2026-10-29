import os
import numpy as np
import pandas as pd

np.random.seed(404)
START_DATE = "2023-01-01 00:00:00"
NUM_DAYS   = 180
FREQ       = "15min"
OUT_DIR  = "data"
os.makedirs(OUT_DIR, exist_ok=True)
RAW_FILE = os.path.join(OUT_DIR, "Technology_Tower_AMI_Export.csv")

periods    = NUM_DAYS * 96
time_index = pd.date_range(start=START_DATE, periods=periods, freq=FREQ)
hours, minutes, dow, month = np.asarray(time_index.hour), np.asarray(time_index.minute), np.asarray(time_index.day_of_week), np.asarray(time_index.month)
dayfrac = hours + (minutes / 60.0)

server_kw = np.clip(120.0 + np.random.normal(0, 2.5, periods) + np.exp(-((dayfrac - 16.0) ** 2) / (2 * 3.0 ** 2)) * 15.0, 100.0, None)
hvac_kw = np.clip((45.0 + 110.0 * ((dayfrac >= 8.0) & (dayfrac <= 19.0)).astype(float)) * np.where(dow >= 5, 0.15, 1.0) + np.random.normal(0, 3.0, periods), 15.0, None)
elevator_kw = (5.0 + 25.0 * (np.exp(-((dayfrac - 9.0) ** 2) / (2 * 0.8 ** 2)) + np.exp(-((dayfrac - 13.0) ** 2) / (2 * 1.0 ** 2)) + np.exp(-((dayfrac - 18.0) ** 2) / (2 * 1.2 ** 2)))) * np.where(dow >= 5, 0.2, 1.0) + np.random.normal(0, 1.0, periods)
lighting_kw = np.clip((15.0 + 35.0 * ((dayfrac >= 8.5) & (dayfrac <= 19.5)).astype(float)) * np.where(dow >= 5, 0.1, 1.0) + np.random.normal(0, 1.0, periods), 5.0, None)
research_lab_kw = np.clip((20.0 + np.random.normal(0, 1.0, periods) + np.exp(-((dayfrac - 22.0) ** 2) / (2 * 3.0 ** 2)) * 30.0) * np.where(dow >= 5, 0.6, 1.0) + np.random.normal(0, 1.0, periods), 10.0, None)
water_pump_kw = np.clip((((dayfrac >= 6.0) & (dayfrac <= 7.0)).astype(float) + ((dayfrac >= 21.0) & (dayfrac <= 22.0)).astype(float)) * 40.0 + np.random.normal(0, 0.5, periods), 0.0, None)
auditorium_kw = np.clip(np.where(((dow == 2) | (dow == 4)) & ((dayfrac >= 14.0) & (dayfrac <= 17.0)), 45.0, 2.0) + np.random.normal(0, 1.0, periods), 0.0, None)

aggregate_kw = np.clip(server_kw + hvac_kw + elevator_kw + lighting_kw + research_lab_kw + water_pump_kw + auditorium_kw + np.random.normal(0, 2.0, periods), 80.0, None)
voltage_v = np.clip(415.0 + np.random.normal(0, 2.0, periods) - 4.0 * (aggregate_kw / aggregate_kw.max()), 400, 425)
power_factor = np.clip(0.92 - (0.08 * (aggregate_kw / aggregate_kw.max())) + np.random.normal(0, 0.015, periods), 0.82, 0.99)
current_a = np.clip((aggregate_kw * 1000.0) / (1.732 * voltage_v * power_factor), 150.0, None)

n_spike = 12
spike_idx = np.random.choice(periods, size=n_spike, replace=False)
aggregate_kw[spike_idx] += np.random.uniform(60.0, 120.0, n_spike)

quality_flag = np.array(["GOOD"] * periods, dtype=object)
df = pd.DataFrame({
    "Meter_Serial_No": "MH-HTC-882194", "Consumer_No": "HT-COMM-50912", "Connection_Type": "HT-Commercial", "Tariff_Category": "HT-I Commercial (Technology)",
    "Reading_Time": time_index, "Active_Energy_kW": np.round(aggregate_kw, 2), "Server_Room_kW": np.round(server_kw, 2),
    "HVAC_kW": np.round(hvac_kw, 2), "Elevators_kW": np.round(elevator_kw, 2), "Office_Light_kW": np.round(lighting_kw, 2),
    "Research_Lab_kW": np.round(research_lab_kw, 2), "Water_Pump_kW": np.round(water_pump_kw, 2), "Auditorium_kW": np.round(auditorium_kw, 2),
    "Voltage_R_Phase": np.round(voltage_v, 1), "Current_R_Phase": np.round(current_a, 1), "Power_Factor": np.round(power_factor, 3)
})
df.to_csv(RAW_FILE, index=False)
print(f"[OK] Tech Tower Data Generated: {RAW_FILE}")