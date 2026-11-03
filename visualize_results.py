from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs")

pred = pd.read_csv(OUT / "real_data_test_predictions.csv", parse_dates=["timestamp"])
metrics = pd.read_csv(OUT / "real_data_model_comparison.csv")
forecast = pd.read_csv(OUT / "real_data_next_6h_forecast.csv", parse_dates=["timestamp"])
history = pd.read_csv(OUT / "real_data_continuous_15min.csv", parse_dates=["timestamp"])

plt.figure(figsize=(13, 5))
plt.plot(pred.timestamp, pred.Actual, label="Actual", linewidth=2)
for col in ["SARIMA", "ARIMA", "Seasonal Naive"]:
    plt.plot(pred.timestamp, pred[col], label=col)
plt.title("Actual vs Predicted Occupancy (6-hour holdout)")
plt.xlabel("Time")
plt.ylabel("Occupancy level")
plt.legend()
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(metrics["Model"], metrics["RMSE"])
plt.title("Model Comparison - RMSE (lower is better)")
plt.ylabel("RMSE")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150)
plt.show()

history["hour"] = history.timestamp.dt.hour
profile = history.groupby("hour")["occupancy"].mean()
plt.figure(figsize=(10, 5))
plt.plot(profile.index, profile.values, marker="o")
plt.title("Average Occupancy by Hour (Selected Continuous Segment)")
plt.xlabel("Hour of day")
plt.ylabel("Average occupancy level")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "hourly_occupancy_pattern.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(forecast.timestamp, forecast.forecast, marker="o")
plt.title("Next 6-Hour Forecast (15-minute intervals)")
plt.xlabel("Future time")
plt.ylabel("Predicted occupancy level")
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(OUT / "next_6h_forecast.png", dpi=150)
plt.show()

print("All visualizations saved in outputs/.")
