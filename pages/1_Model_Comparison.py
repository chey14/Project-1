import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("🤖 Forecast Model Comparison")
st.caption("Transformer vs XGBoost vs Persistence baseline on the same 165-day test period")

try:
    tf_df = pd.read_csv("results_dashboard/forecast_results.csv", parse_dates=["Time"])
    xg_df = pd.read_csv("results_dashboard/xgb_results.csv", parse_dates=["Time"])
    with open("results_dashboard/forecast_metrics.json") as f: tf_m = json.load(f)
    with open("results_dashboard/xgb_metrics.json") as f: xg_m = json.load(f)
except FileNotFoundError:
    st.error("⚠️ Missing result files. Run 5_transformer_forecast.py and 8_xgboost_forecast.py first.")
    st.stop()

df = tf_df.rename(columns={"Predicted": "Transformer"}).merge(
    xg_df.drop(columns=["Actual"]), on="Time", how="inner")

models = {
    "Transformer": tf_m,
    "XGBoost": xg_m["XGBoost"],
    "XGBoost + Calendar": xg_m["XGBoost + Calendar"],
    "Persistence": xg_m["Persistence"],
}
col_map = {"Transformer": "Transformer", "XGBoost": "XGB",
           "XGBoost + Calendar": "XGB_Calendar", "Persistence": "Persistence"}

# ---- metric table ----
mt = pd.DataFrame({name: {"MAE (W)": m["MAE"], "RMSE (W)": m["RMSE"], "MAPE (%)": m["MAPE"]}
                   for name, m in models.items()}).T
best = mt["MAE (W)"].idxmin()
c1, c2, c3 = st.columns(3)
c1.metric("Best model (by MAE)", best)
c2.metric("Transformer MAE", f"{tf_m['MAE']:.2f} W")
c3.metric("XGBoost MAE", f"{xg_m['XGBoost']['MAE']:.2f} W")
st.dataframe(mt.style.format("{:.2f}").highlight_min(axis=0, color="#b3ffcc"), width='stretch')

# ---- metric bar charts ----
st.subheader("Metrics side by side")
cols = st.columns(3)
for c, metric in zip(cols, ["MAE (W)", "RMSE (W)", "MAPE (%)"]):
    d = mt[metric].reset_index(); d.columns = ["Model", metric]
    c.plotly_chart(px.bar(d, x="Model", y=metric, color="Model", title=metric)
                   .update_layout(height=340, showlegend=False), width='stretch')

# ---- overlay chart ----
st.subheader("Actual vs predicted (test period)")
chosen = st.multiselect("Models to show", list(models.keys()), default=["Transformer", "XGBoost"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Time"], y=df["Actual"], mode="lines", name="Actual", line=dict(color="black", width=2)))
for name in chosen:
    fig.add_trace(go.Scatter(x=df["Time"], y=df[col_map[name]], mode="lines", name=name, line=dict(dash="dash")))
st.plotly_chart(fig.update_layout(height=450), width='stretch')

# ---- absolute error per day ----
st.subheader("Absolute error per day")
fig2 = go.Figure()
for name in chosen:
    fig2.add_trace(go.Scatter(x=df["Time"], y=(df["Actual"] - df[col_map[name]]).abs(), mode="lines", name=name))
st.plotly_chart(fig2.update_layout(height=380, yaxis_title="|error| (W)"), width='stretch')
st.caption("Spikes in this chart line up with the month-boundary steps in the synthetic data.")