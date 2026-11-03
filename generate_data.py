
"""Generate a clearly-labelled synthetic campus occupancy dataset."""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
facilities = {
    "Library": {"capacity": 150, "base": 48, "weekday_peak": 68, "weekend_factor": 0.55},
    "Cafeteria": {"capacity": 220, "base": 25, "weekday_peak": 150, "weekend_factor": 0.45},
    "Computer Lab": {"capacity": 80, "base": 12, "weekday_peak": 48, "weekend_factor": 0.35},
    "Gym": {"capacity": 70, "base": 8, "weekday_peak": 42, "weekend_factor": 0.65},
    "Parking": {"capacity": 250, "base": 35, "weekday_peak": 190, "weekend_factor": 0.30},
}
timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=24*120, freq="h")
rows = []

for facility, cfg in facilities.items():
    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday()
        # Smooth daily demand profile with facility-specific peak periods.
        if facility == "Library":
            daily = 0.20 + 0.75*np.exp(-((hour-16)/4.0)**2)
        elif facility == "Cafeteria":
            daily = 0.08 + 0.95*np.exp(-((hour-13)/2.2)**2) + 0.55*np.exp(-((hour-20)/2.0)**2)
        elif facility == "Computer Lab":
            daily = 0.08 + 0.85*np.exp(-((hour-15)/3.0)**2)
        elif facility == "Gym":
            daily = 0.05 + 0.85*np.exp(-((hour-18)/2.5)**2)
        else:
            daily = 0.05 + 0.90*np.exp(-((hour-10)/2.5)**2) + 0.70*np.exp(-((hour-16)/2.5)**2)
        weekend = cfg["weekend_factor"] if weekday >= 5 else 1.0
        weekly = 0.90 + 0.10*np.sin(2*np.pi*(weekday+1)/7)
        trend = 1 + 0.0008*((ts - timestamps[0]).total_seconds()/3600)
        expected = cfg["base"] + cfg["weekday_peak"]*daily*weekend*weekly*trend
        noise = rng.normal(0, max(2, cfg["capacity"]*0.025))
        occupancy = int(np.clip(round(expected + noise), 0, cfg["capacity"]*1.12))
        rows.append([ts, facility, occupancy, cfg["capacity"]])

df = pd.DataFrame(rows, columns=["timestamp", "facility", "occupancy", "capacity"])
out = Path(__file__).parent / "data" / "campus_occupancy.csv"
df.to_csv(out, index=False)
print(f"Created {len(df):,} rows: {out}")
print("NOTE: This is synthetic demonstration data, not real university sensor data.")
