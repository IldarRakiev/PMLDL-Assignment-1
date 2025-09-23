import streamlit as st
import requests
from feature_ranges import feature_stats

API_URL = "http://api:8000"

st.title("🏡 California Housing Price Predictor (Gradient Boosting)")
st.write("Enter the neighborhood characteristics to predict the median house value:")

# Default values (mean values, in human-readable units)
default_inputs = {
    "Median Income ($ per household)": 38700.0,
    "House Age (years)": 28.6,
    "Average Rooms per Household": 5.43,
    "Average Bedrooms per Household": 1.07,
    "Population (number of residents)": 1425.0,
    "Average Household Size": 3.07,
    "Latitude": 35.6,
    "Longitude": -119.6,
}

# Step sizes
steps = {
    "Median Income ($ per household)": 1000.0,
    "House Age (years)": 1.0,
    "Average Rooms per Household": 0.1,
    "Average Bedrooms per Household": 0.1,
    "Population (number of residents)": 50.0,
    "Average Household Size": 0.1,
    "Latitude": 0.01,
    "Longitude": 0.01,
}

# Map UI field names to dataset feature names
ui_to_feature = {
    "Median Income ($ per household)": "MedInc",
    "House Age (years)": "HouseAge",
    "Average Rooms per Household": "AveRooms",
    "Average Bedrooms per Household": "AveBedrms",
    "Population (number of residents)": "Population",
    "Average Household Size": "AveOccup",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

inputs = []
cols = st.columns(2)

for i, (name, default) in enumerate(default_inputs.items()):
    feature = ui_to_feature[name]
    fmin = feature_stats[feature]["min"]
    fmax = feature_stats[feature]["max"]

    # Special case: MedInc is in $10,000s in the dataset → scale to dollars for UI
    if feature == "MedInc":
        fmin, fmax = fmin * 10000, fmax * 10000

    with cols[i % 2]:
        val = st.number_input(
            name,
            value=float(default),
            min_value=fmin,
            max_value=fmax,
            step=steps[name],
        )
        inputs.append(val)

# Convert user inputs to model format
processed_inputs = inputs.copy()
processed_inputs[0] = processed_inputs[0] / 10000  # MedInc back to tens of thousands

if st.button("🔮 Predict"):
    payload = {"features": processed_inputs}
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        st.success(f"Predicted Median House Value: ${data['prediction'] * 100_000:,.0f}")
    except Exception as e:
        st.error(f"Error contacting API: {e}")
