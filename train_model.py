"""Train and evaluate one ARIMA model per facility; export metrics and forecasts."""
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/campus_occupancy.csv")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--test-hours", type=int, default=48)
    parser.add_argument("--order", nargs=3, type=int, default=[2, 1, 2], metavar=("P", "D", "Q"))
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    required = {"timestamp", "facility", "occupancy", "capacity"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain: {sorted(required)}")

    metrics, forecasts = [], []
    for facility, part in df.groupby("facility"):
        s = (part.set_index("timestamp")["occupancy"].sort_index()
             .resample("h").mean().interpolate(limit_direction="both"))
        if len(s) <= args.test_hours + 10:
            print(f"Skipping {facility}: insufficient observations")
            continue
        train, test = s.iloc[:-args.test_hours], s.iloc[-args.test_hours:]
        try:
            validation_model = ARIMA(train, order=tuple(args.order)).fit()
            validation_pred = validation_model.forecast(len(test))
            mae = mean_absolute_error(test, validation_pred)
            rmse = np.sqrt(mean_squared_error(test, validation_pred))

            final_model = ARIMA(s, order=tuple(args.order)).fit()
            future = final_model.get_forecast(args.horizon)
            values = future.predicted_mean.clip(lower=0)
            capacity = float(part["capacity"].median())
            metrics.append({"facility": facility, "observations": len(s),
                            "ARIMA_order": str(tuple(args.order)), "MAE": mae, "RMSE": rmse,
                            "capacity": capacity})
            for timestamp, value in values.items():
                forecasts.append({"timestamp": timestamp, "facility": facility,
                                  "predicted_occupancy": float(value), "capacity": capacity,
                                  "over_capacity": bool(value >= capacity)})
            print(f"{facility}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        except Exception as exc:
            print(f"Could not train {facility}: {exc}")

    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv("outputs/model_metrics.csv", index=False)
    pd.DataFrame(forecasts).to_csv("outputs/future_forecasts.csv", index=False)
    print("Saved outputs/model_metrics.csv and outputs/future_forecasts.csv")

if __name__ == "__main__":
    main()
