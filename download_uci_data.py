"""Download UCI Room Occupancy Estimation (ID 864) and convert to dashboard schema."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

print("Fetching UCI Room Occupancy Estimation dataset (ID 864)...")
dataset = fetch_ucirepo(id=864)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
df = pd.concat([X, y], axis=1)

# Original UCI fields are Date, Time and Room_Occupancy_Count.
df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)
df["occupancy"] = pd.to_numeric(df["Room_Occupancy_Count"], errors="coerce")
df = df.dropna(subset=["timestamp", "occupancy"]).sort_values("timestamp")

# This is a single controlled room, not five campus facilities.
# Capacity=3 is the maximum occupancy level represented in this experimental dataset,
# NOT a building/fire-code capacity. Change it only if you have a verified room capacity.
df["facility"] = "UCI Experimental Room"
df["capacity"] = 3

# Preserve raw sensor data and create a dashboard-compatible table.
out = df[["timestamp", "facility", "occupancy", "capacity"]].copy()
out.to_csv(DATA / "campus_occupancy.csv", index=False)
df.to_csv(DATA / "uci_room_occupancy_raw.csv", index=False)

print(f"Saved {len(out):,} real observations to {DATA / 'campus_occupancy.csv'}")
print(f"Date range: {out.timestamp.min()} to {out.timestamp.max()}")
print(f"Observed occupancy values: {sorted(out.occupancy.unique().tolist())}")
print("IMPORTANT: UCI data is from one controlled 6m x 4.6m room, not a campus.")
