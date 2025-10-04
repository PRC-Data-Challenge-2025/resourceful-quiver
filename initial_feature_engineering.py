# %%

## This is the code for initial feature engineering

## First check that we have all the data
import pandas as pd
import os

# Read the fuel parquet
df = pd.read_parquet("data/fuel_train.parquet")

# Get unique flight IDs from the dataset
unique_ids = set(df["flight_id"].unique())
print("Number of unique flight IDs in fuel_train:", len(unique_ids))

# List all IDs from filenames in flight_train folder
folder_path = "data/flight_train/"
ids = {os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith(".parquet")}
print("Total IDs from filenames:", len(ids))

# Compare
available_ids = ids & unique_ids        # present in both
missing_ids = ids - unique_ids          # exist in folder but not in fuel_train
extra_ids = unique_ids - ids            # exist in fuel_train but no file in folder

print("Available IDs:", len(available_ids))
print("Missing IDs:", len(missing_ids))
print("Extra IDs:", len(extra_ids))

assert((len(missing_ids) == 0) & (len(extra_ids) == 0))

## Then check we have all the airport in our dataset
flightlist_train = pd.read_parquet("data/flightlist_train.parquet")
df_airport = pd.read_parquet("data/apt.parquet")

# Combine origin and destination ICAO codes into one unique set
icao_codes = set(flightlist_train['origin_icao']).union(set(flightlist_train['destination_icao']))

# All airport identifiers
airport_idents = set(df_airport['icao'])

# Differences
missing_in_airports = icao_codes - airport_idents   # Codes in flights but not in airports.csv
unused_in_flights = airport_idents - icao_codes    # Codes in airports.csv but not used in flights

print(f"Missing in airports.csv: {len(missing_in_airports)}")

assert(len(missing_in_airports) == 0)

## Add the airport lat lon to the flightlist

# Select only needed columns
df_airport = df_airport[['icao', 'latitude', 'longitude', 'elevation']]

# Merge for origin
flightlist_train = flightlist_train.merge(
    df_airport,
    left_on='origin_icao',
    right_on='icao',
    how='left',
    suffixes=('', '_origin')
)

# Rename merged columns for clarity
flightlist_train.rename(columns={
    'latitude': 'origin_lat',
    'longitude': 'origin_lon',
    'elevation': 'origin_elev'
}, inplace=True)

# Drop the duplicate ident column
flightlist_train.drop(columns=['icao'], inplace=True)

# Merge for destination
flightlist_train = flightlist_train.merge(
    df_airport,
    left_on='destination_icao',
    right_on='icao',
    how='left',
    suffixes=('', '_dest')
)

# Rename merged columns for destination
flightlist_train.rename(columns={
    'latitude': 'dest_lat',
    'longitude': 'dest_lon',
    'elevation': 'dest_elev'
}, inplace=True)

# Drop the duplicate ident column
flightlist_train.drop(columns=['icao'], inplace=True)

output_path = "data/flightlist_train_latlon.parquet"
flightlist_train.to_parquet(output_path, index=False)

# %%
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

# Paths
FLIGHT_PATH = "data/flight_train/"

# Read fuel data
df_fuel = pd.read_parquet("data/fuel_train.parquet")

# Convert timestamps
df_fuel["start"] = pd.to_datetime(df_fuel["start"])
df_fuel["end"] = pd.to_datetime(df_fuel["end"])

# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Optional: subset for quick testing
# df_fuel = df_fuel[:1]

# %%

# ---- function to process a single fuel_row ----
def process_flight_row(fuel_row):
    flight_id = fuel_row["flight_id"]
    start_time = fuel_row["start"]
    end_time = fuel_row["end"]
    fuel_burnt_kg = fuel_row["fuel_kg"]

    file_path = os.path.join(FLIGHT_PATH, f"{flight_id}.parquet")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file for flight_id: {flight_id}")
        return None

    # Read the flight parquet
    df_sub = pd.read_parquet(file_path)
    df_sub["timestamp"] = pd.to_datetime(df_sub["timestamp"])

    # Find closest rows by timestamp
    start_row = df_sub.iloc[(df_sub["timestamp"] - start_time).abs().argmin()]
    end_row = df_sub.iloc[(df_sub["timestamp"] - end_time).abs().argmin()]

    # Compute Haversine distance
    distance_km = haversine(
        start_row["latitude"], start_row["longitude"],
        end_row["latitude"], end_row["longitude"]
    )

    return {
        "flight_id": flight_id,
        "start_time": start_time,
        "end_time": end_time,
        "distance_km": distance_km,
        "fuel_kg": fuel_burnt_kg,
        "typecode": df_sub["typecode"].unique()[0],
    }


# ---- Run in parallel ----
results = Parallel(n_jobs=3, verbose=10)(
    delayed(process_flight_row)(row) for _, row in df_fuel.iterrows()
)

# Filter out None results (missing files, errors)
distances = [r for r in results if r is not None]

# Convert to DataFrame
df_distance = pd.DataFrame(distances)
print(df_distance.head())

# %%
import matplotlib.pyplot as plt

# Get unique aircraft types
types = df_distance['typecode'].unique()

# Create subplots
fig, axes = plt.subplots(len(types), 1, figsize=(8, 5 * len(types)), sharex=True, sharey=True)

# If there's only one type, axes may not be an array
if len(types) == 1:
    axes = [axes]

# Plot each type separately
for ax, t in zip(axes, types):
    subset = df_distance[df_distance['typecode'] == t]
    ax.scatter(subset['distance_km'], subset['fuel_kg'], label=t)
    ax.set_title(f"{t} — Distance vs Fuel Burn")
    ax.set_xlim(-50, 1250)
    ax.set_ylim(-200, 12500)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Fuel Burn (kg)")
    ax.legend()

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

# Number of random flights to plot
N = min(20, len(df_fuel))  # adjust as needed
sample_rows = df_fuel.sample(n=N, random_state=42)

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

counter = 0

for _, fuel_row in sample_rows.iterrows():
    counter += 1
    flight_id = fuel_row["flight_id"]
    start_time = fuel_row["start"]
    end_time = fuel_row["end"]

    file_path = os.path.join(FLIGHT_PATH, f"{flight_id}.parquet")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file for {flight_id}")
        continue

    # --- Read the flight data ---
    df_sub = pd.read_parquet(file_path)
    df_sub["timestamp"] = pd.to_datetime(df_sub["timestamp"])

    # Find closest start/end indices
    start_idx = (df_sub["timestamp"] - start_time).abs().argmin()
    end_idx = (df_sub["timestamp"] - end_time).abs().argmin()

    start_point = df_sub.iloc[start_idx]
    end_point = df_sub.iloc[end_idx]

    # --- Create subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"Flight {flight_id} — {df_sub['typecode'].iloc[0]}", fontsize=12, y=0.96)

    # --- Plot 1: Flight path (Lat vs Lon) ---
    for src, sub_src in df_sub.groupby("source"):
        color = "red" if src.lower() == "acars" else "blue"
        ax1.plot(sub_src["longitude"], sub_src["latitude"], color=color, alpha=0.5, linewidth=1.2, label=src.upper())

    # Start (black) and End (magenta)
    ax1.scatter(start_point["longitude"], start_point["latitude"], color="black", s=50, zorder=5, label="Start")
    ax1.scatter(end_point["longitude"], end_point["latitude"], color="#ff00ff", s=50, zorder=5, label="End")

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Flight Path (Blue = ADSB, Red = ACARS)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Plot 2: Altitude vs Time ---
    for src, sub_src in df_sub.groupby("source"):
        color = "red" if src.lower() == "acars" else "blue"
        ax2.plot(sub_src["timestamp"], sub_src["altitude"], color=color, alpha=0.5, linewidth=1.2, label=src.upper())

    ax2.axvline(start_point["timestamp"], color="black", linestyle="--", label="Start")
    ax2.axvline(end_point["timestamp"], color="#ff00ff", linestyle="--", label="End")

    ax2.set_xlabel("Timestamp (UTC)")
    ax2.set_ylabel("Altitude (ft or m)")
    ax2.set_title("Altitude vs Time")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Format timestamps on x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # Add note about timestamp format
    ax2.text(
        0.99, -0.25,
        "Timestamp format: YYYY-MM-DD HH:MM:SS (UTC)",
        transform=ax2.transAxes,
        ha="right", va="center",
        fontsize=9, color="gray"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(f"figures/{counter:03d}_randsample.png", dpi=150)
    plt.show()
