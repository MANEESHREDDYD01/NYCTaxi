import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)  # Corrected '_file_' to '__file__'
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

st.title("Mean Absolute Error (MAE) by Pickup Hour")

# Sidebar for user input
st.sidebar.header("Settings")

# Load location lookup table
lookup_path = Path(__file__).parent / "taxi-zone-lookup.csv"
lookup_df = pd.read_csv(lookup_path)

# Ensure correct column names
lookup_df.rename(columns={"LocationID": "pickup_location_id", "Borough": "borough", "Zone": "location_name"}, inplace=True)

# Dropdown to select location by name
selected_location = st.sidebar.selectbox("Select Pickup Location", lookup_df["location_name"].unique())

# Get the corresponding location ID
selected_location_id = lookup_df.loc[lookup_df["location_name"] == selected_location, "pickup_location_id"].values[0]

# Slider to select past hours
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=12,
    step=1,
)

# Fetch data
st.write(f"Fetching data for {selected_location} ({selected_location_id}) for the past {past_hours} hours...")
df1 = fetch_hourly_rides(past_hours)
df2 = fetch_predictions(past_hours)

# Ensure both DataFrames have the 'pickup_location_id' and 'pickup_hour' columns
if "pickup_location_id" in df1.columns and "pickup_location_id" in df2.columns:
    # Merge data on 'pickup_location_id' and 'pickup_hour'
    merged_df = pd.merge(df1, df2, on=["pickup_location_id", "pickup_hour"], how="inner")
else:
    st.error("Column 'pickup_location_id' is missing from one of the data sources! Check fetch_hourly_rides and fetch_predictions functions.")
    st.stop()

# Filter data based on selected location
filtered_df = merged_df[merged_df["pickup_location_id"] == selected_location_id]

# Ensure required columns exist before computing absolute error
if "predicted_demand" not in filtered_df.columns or "rides" not in filtered_df.columns:
    st.error("Missing required columns in the merged data! Check fetch_predictions and fetch_hourly_rides sources.")
    st.stop()

# Compute absolute error
filtered_df["absolute_error"] = abs(filtered_df["predicted_demand"] - filtered_df["rides"])

# Group by pickup_hour and calculate MAE
mae_by_hour = filtered_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Create a Plotly line plot
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for {selected_location} in the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display the plot
st.plotly_chart(fig)
st.write(f'Average MAE for {selected_location}: {mae_by_hour["MAE"].mean()}')
