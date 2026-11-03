import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Campus Facility Overload Prediction", page_icon="🏫", layout="wide")
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"

st.title("🏫 Campus Facility Overload Prediction")
st.caption("Experimental occupancy forecasting prototype using ARIMA and SARIMA")

required = [
    OUT / "real_data_continuous_15min.csv",
    OUT / "real_data_test_predictions.csv",
    OUT / "real_data_model_comparison.csv",
    OUT / "real_data_next_6h_forecast.csv"
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    st.error("Run `python train_real_data.py` first. Missing files: " + ", ".join(missing))
    st.stop()

history = pd.read_csv(required[0], parse_dates=["timestamp"])
pred = pd.read_csv(required[1], parse_dates=["timestamp"])
metrics = pd.read_csv(required[2])
forecast = pd.read_csv(required[3], parse_dates=["timestamp"])

st.sidebar.header("Dashboard Controls")
st.sidebar.write("**Dataset:** UCI Room Occupancy Estimation")
st.sidebar.write("**Selected segment:** longest continuous segment")
st.sidebar.write("**Forecast horizon:** 6 hours")
st.sidebar.warning("One experimental room only. This is not live campus data. The maximum observed label of 3 is not an official room capacity.")

latest = float(history["occupancy"].iloc[-1])
maximum = float(history["occupancy"].max())
average = float(history["occupancy"].mean())

a, b, c, d = st.columns(4)
a.metric("Latest observed level", f"{latest:.2f}")
b.metric("Maximum observed level", f"{maximum:.2f}")
c.metric("Average observed level", f"{average:.2f}")
d.metric("Test observations", f"{len(pred)}")

st.subheader("Observed Occupancy History")
st.caption("Only the longest continuous source-data segment is shown. No interpolation across long gaps.")
fig = px.line(history, x="timestamp", y="occupancy", labels={"timestamp":"Time", "occupancy":"Occupancy level"})
fig.update_layout(height=400)
st.plotly_chart(fig, width="stretch")

st.subheader("Actual vs Predicted Occupancy")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred.timestamp, y=pred.Actual, name="Actual", mode="lines", line=dict(width=3)))
for col in ["SARIMA", "ARIMA", "Seasonal Naive"]:
    if col in pred.columns:
        fig.add_trace(go.Scatter(x=pred.timestamp, y=pred[col], name=col, mode="lines"))
fig.update_layout(height=430, xaxis_title="Time", yaxis_title="Occupancy level", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.subheader("Model Performance")
st.table(metrics, use_container_width=True)
if "RMSE" in metrics.columns:
    fig = px.bar(metrics, x="Model", y="RMSE", title="RMSE comparison (lower is better)", text_auto=".3f")
    st.plotly_chart(fig, width="stretch")

st.subheader("Average Occupancy by Hour")
history["hour"] = history.timestamp.dt.hour
profile = history.groupby("hour", as_index=False)["occupancy"].mean()
fig = px.bar(profile, x="hour", y="occupancy", labels={"hour":"Hour of day", "occupancy":"Average occupancy level"})
fig.update_xaxes(dtick=1)
st.plotly_chart(fig, width="stretch")

st.subheader("Next 6-Hour Forecast")
fig = px.line(forecast, x="timestamp", y="forecast", markers=True, labels={"timestamp":"Future time", "forecast":"Predicted occupancy level"})
st.plotly_chart(fig, width="stretch")
st.table(forecast, use_container_width=True)

with st.expander("View selected continuous dataset"):
    st.table(history, use_container_width=True)
with st.expander("View test predictions"):
    st.table(pred, use_container_width=True)

st.caption("Academic prototype using UCI experimental-room sensor data. Not a campus deployment or safety-capacity system.")
