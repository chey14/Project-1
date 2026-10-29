import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")
st.title("⚡ Smart Household Energy Dashboard")
st.caption("Synthetic AMI export • Transformer forecasting • Isolation Forest anomaly detection • Appliance & Power-Quality analytics")

@st.cache_data
def load_all():
    files = {"anomaly": "results_dashboard/anomaly_results.csv", "forecast": "results_dashboard/forecast_results.csv", "metrics": "results_dashboard/forecast_metrics.json", "utility": "results_dashboard/utility_results.json"}
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing: st.error("⚠️ Missing result files. Run the scripts first."); st.stop()
    anomaly = pd.read_csv(files["anomaly"], parse_dates=["Time"])
    forecast = pd.read_csv(files["forecast"], parse_dates=["Time"])
    with open(files["metrics"]) as f: metrics = json.load(f)
    with open(files["utility"]) as f: utils = json.load(f)
    return anomaly, forecast, metrics, utils

anomaly_df, forecast_df, metrics, utils = load_all()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Consumption", f"{utils['total_kwh']:.1f} kWh")
c2.metric("Avg Monthly Bill", f"₹ {pd.Series(utils['monthly_bill']).mean():.0f}")
c3.metric("Anomalies Flagged", int(anomaly_df["Anomaly"].sum()))
c4.metric("Forecast MAPE", f"{metrics['MAPE']:.2f}%")
c5.metric("Power Quality", f"{utils['power_quality_index']}/100")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "🚨 Anomalies", "🕐 Usage", "🔌 Appliances", "📊 Quality & Bill"])

with tab1:
    st.subheader("Next-Day Energy Forecast (Transformer)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["Time"], y=forecast_df["Actual"], mode="lines", name="Actual", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=forecast_df["Time"], y=forecast_df["Predicted"], mode="lines", name="Predicted", line=dict(color="orange", dash="dash")))
    st.plotly_chart(fig.update_layout(height=420), width='stretch')

with tab2:
    st.subheader("Isolation Forest — 21-Minute Contextual Anomalies")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anomaly_df["Time"], y=anomaly_df["Center_Power"], mode="lines", name="Aggregate Power", line=dict(color="lightgray")))
    a = anomaly_df[anomaly_df["Anomaly"] == 1]
    fig.add_trace(go.Scatter(x=a["Time"], y=a["Center_Power"], mode="markers", name="Anomaly", marker=dict(color="red", size=8)))
    st.plotly_chart(fig.update_layout(height=420), width='stretch')

with tab3:
    st.subheader("Average Power by Hour of Day")
    hdf = pd.DataFrame({"Hour": list(utils["hour_profile"].keys()), "Avg kW": list(utils["hour_profile"].values())}).sort_values("Hour")
    st.plotly_chart(px.bar(hdf, x="Hour", y="Avg kW", color="Avg kW", color_continuous_scale="Teal").update_layout(height=380), width='stretch')
    st.subheader("Average Power by Day of Week")
    ddf = pd.DataFrame({"Day": list(utils["day_profile"].keys()), "Avg kW": list(utils["day_profile"].values())})
    st.plotly_chart(px.bar(ddf, x="Day", y="Avg kW", color="Avg kW", color_continuous_scale="Oranges").update_layout(height=380), width='stretch')

with tab4:
    st.subheader("Appliance Energy Contribution")
    adf = pd.DataFrame({"Appliance": list(utils["appliance_totals_kwh"].keys()), "kWh": list(utils["appliance_totals_kwh"].values()), "Share (%)": list(utils["appliance_share_pct"].values())})
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(adf, names="Appliance", values="kWh", title="Total energy share").update_layout(height=400), width='stretch')
    c2.plotly_chart(px.bar(adf, x="Appliance", y="Share (%)", color="Appliance", title="% of household energy").update_layout(height=400), width='stretch')
    
    st.divider()
    st.subheader("Hourly Appliance Usage Profile")
    raw_df = pd.read_csv("data/House_1_smartmeter_raw.csv")
    raw_df['Hour'] = raw_df['Timestamp'].str[11:13].astype(int)
    hourly_avg = raw_df.groupby('Hour')[["Fridge_kW", "AC_kW", "WaterHeater_kW", "Lighting_kW"]].mean().reset_index()
    melted = hourly_avg.melt(id_vars="Hour", var_name="Appliance", value_name="Avg kW")
    st.plotly_chart(px.bar(melted, x="Hour", y="Avg kW", color="Appliance", barmode="group").update_layout(height=400, xaxis=dict(tickmode='linear', tick0=0, dtick=2)), width='stretch')
    
    st.divider()
    st.subheader("Appliance Health Monitor (Z-Score)")
    health_df = pd.DataFrame([{"Appliance": k, "Recent 24h (kW)": v["mean_kW"], "Baseline (kW)": v["baseline_kW"], "Z-score": v["z_score"], "Status": v["status"]} for k, v in utils["appliance_health"].items()])
    st.dataframe(health_df.style.format({"Recent 24h (kW)": "{:.3f}", "Baseline (kW)": "{:.3f}", "Z-score": "{:+.2f}"}), width='stretch')

with tab5:
    st.subheader("Power Quality Index")
    b = utils["pqi_breakdown"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Voltage score", f"{b['voltage_score']}/100"); c2.metric("Power-factor score", f"{b['powerfactor_score']}/100"); c3.metric("Frequency score", f"{b['frequency_score']}/100")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=utils["power_quality_index"], gauge={"axis": {"range": [0, 100]}, "bar": {"color": "teal"}, "steps": [{"range": [0, 50], "color": "#ffb3b3"}, {"range": [50, 75], "color": "#fff2b3"}, {"range": [75, 100], "color": "#b3ffcc"}]}, title={"text": "Power Quality Index"}))
    st.plotly_chart(fig.update_layout(height=320), width='stretch')
    
    st.subheader("Monthly Consumption & Estimated Bill")
    bill_df = pd.DataFrame({"Month": list(utils["monthly_kwh"].keys()), "kWh": list(utils["monthly_kwh"].values()), "₹ Bill": list(utils["monthly_bill"].values())})
    bill_df["Month"] = pd.to_datetime(bill_df["Month"]).dt.strftime("%b %Y")
    st.dataframe(bill_df, width='stretch')