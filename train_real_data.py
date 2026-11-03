import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "campus_occupancy.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

print("Loading original UCI room occupancy observations...")
raw = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
raw = raw[["timestamp", "occupancy"]].dropna().sort_values("timestamp")
raw = raw.drop_duplicates(subset="timestamp").set_index("timestamp")

gap_hours = raw.index.to_series().diff().dt.total_seconds().div(3600)
segment_ids = gap_hours.gt(2).cumsum()
segments = [part for _, part in raw.groupby(segment_ids)]
selected = max(segments, key=len)

print(f"Selected continuous segment: {selected.index.min()} to {selected.index.max()}")
print(f"Raw readings: {len(selected)}")

series = selected["occupancy"].resample("15min").mean().asfreq("15min")
series = series.interpolate(method="time", limit=1, limit_area="inside").dropna()

test_size = 72
seasonal_period = 96

if len(series) - test_size <= seasonal_period:
    raise ValueError("Not enough training observations for a valid 24-hour seasonal-naive comparison.")

train = series.iloc[:-test_size]
test = series.iloc[-test_size:]

print(f"15-minute observations: {len(series)}")
print(f"Train: {len(train)} | Test: {len(test)}")
print(f"Test period: {test.index.min()} to {test.index.max()}")
print("Test occupancy distribution:")
print(test.round(3).value_counts().sort_index().to_string())

print("\nFitting ARIMA(2,1,2)...")
arima_fit = ARIMA(train, order=(2, 1, 2)).fit()
arima_pred = np.asarray(arima_fit.forecast(steps=test_size))

print("Fitting exploratory SARIMA...")
sarima_fit = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, seasonal_period),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False, maxiter=200)
sarima_pred = np.asarray(sarima_fit.forecast(steps=test_size))

# For each test timestamp, use the value 24 hours earlier.
naive_pred = np.array([
    series.iloc[len(train) + i - seasonal_period]
    for i in range(test_size)
], dtype=float)

actual = test.to_numpy()
preds = {
    "ARIMA": np.clip(arima_pred, 0, 3),
    "SARIMA": np.clip(sarima_pred, 0, 3),
    "Seasonal Naive": np.clip(naive_pred, 0, 3)
}

prediction_df = pd.DataFrame({
    "timestamp": test.index,
    "Actual": actual,
    "ARIMA": preds["ARIMA"],
    "SARIMA": preds["SARIMA"],
    "Seasonal Naive": preds["Seasonal Naive"]
})

occupied = actual > 0
results = []
for model_name, values in preds.items():
    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(actual, values),
        "RMSE": np.sqrt(mean_squared_error(actual, values)),
        "MAE_when_occupied": (
            mean_absolute_error(actual[occupied], values[occupied])
            if occupied.any() else np.nan
        )
    })

metrics = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
best_model = metrics.loc[0, "Model"]

print("\nMODEL COMPARISON (18-hour chronological holdout)")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nBest model by holdout RMSE: {best_model}")

prediction_df.to_csv(OUT / "real_data_test_predictions.csv", index=False)
metrics.to_csv(OUT / "real_data_model_comparison.csv", index=False)

print("\nGenerating next 6-hour forecast...")
if best_model == "ARIMA":
    final_fit = ARIMA(series, order=(2, 1, 2)).fit()
    future = np.asarray(final_fit.forecast(steps=24))
elif best_model == "SARIMA":
    final_fit = SARIMAX(
        series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False, maxiter=200)
    future = np.asarray(final_fit.forecast(steps=24))
else:
    # Forecast the next six hours using the same six-hour window one day ago.
    future = series.iloc[-seasonal_period:-seasonal_period + 24].to_numpy()

future_index = pd.date_range(
    start=series.index[-1] + pd.Timedelta(minutes=15),
    periods=24,
    freq="15min"
)

forecast_df = pd.DataFrame({
    "timestamp": future_index,
    "forecast": np.clip(future, 0, 3),
    "selected_model": best_model
})
forecast_df.to_csv(OUT / "real_data_next_6h_forecast.csv", index=False)

history_df = series.rename("occupancy").to_frame()
history_df["facility"] = "UCI Experimental Room"
history_df["capacity"] = 3
history_df.to_csv(OUT / "real_data_continuous_15min.csv", index_label="timestamp")

print("\nTRAINING COMPLETED")
print("Saved updated model comparison, test predictions, forecast, and continuous series.")
print("Forecast interval: 15 minutes | Forecast horizon: 6 hours")
print("Caution: one experimental room only; not campus-wide occupancy data.")
