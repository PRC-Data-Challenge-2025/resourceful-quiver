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

## Now do the plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Filter rows with valid coordinates
flights_valid = flightlist_train.dropna(subset=['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'])

# Create map
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)
ax.set_global()

# Plot flight paths
for _, row in flights_valid.iterrows():
    ax.plot(
        [row['origin_lon'], row['dest_lon']],
        [row['origin_lat'], row['dest_lat']],
        color='red',
        linewidth=0.5,
        alpha=0.3,
        transform=ccrs.Geodetic()  # ensures great-circle projection
    )

plt.title("Flight Origins and Destinations (Great Circle)", fontsize=16)
plt.show()