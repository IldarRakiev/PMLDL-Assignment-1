import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from feature_ranges import feature_stats
import numpy as np

API_URL = "http://api:8000"


data = fetch_california_housing(as_frame=True)
X = data.frame

st.set_page_config(page_title="California Housing Price Predictor", layout="wide")
st.title("🏡 California Housing Price Predictor")
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

help_texts = {
    "Median Income ($ per household)": "Median income of households in the block (in US dollars, "
                                        "e.g., 38700 means $38,700). Internally converted to tens of thousands.",
    "House Age (years)": "Median age of houses in the block (in years).",
    "Average Rooms per Household": "Average number of rooms per household in the block.",
    "Average Bedrooms per Household": "Average number of bedrooms per household in the block.",
    "Population (number of residents)": "Total population of the block (number of residents).",
    "Average Household Size": "Average number of people living in each household in the block.",
    "Latitude": "Geographical latitude of the block (approx. 32 to 42 for California).",
    "Longitude": "Geographical longitude of the block (approx. -124 to -114 for California).",
}


tabs = st.tabs(["🔮 Prediction", "📊 Feature Distribution"])


with tabs[0]:
    st.subheader("Make a Prediction")

    inputs = []
    cols = st.columns(2)

    for i, (name, default) in enumerate(default_inputs.items()):
        feature = ui_to_feature[name]
        fmin = feature_stats[feature]["min"]
        fmax = feature_stats[feature]["max"]

        if feature == "MedInc":  # MedInc в датасете в десятках тысяч
            fmin, fmax = fmin * 10000, fmax * 10000

        with cols[i % 2]:
            val = st.number_input(
                name,
                value=float(default),
                min_value=fmin,
                max_value=fmax,
                step=steps[name],
                help=help_texts[name],
                format="%.3f" if steps[name] < 1 else "%.0f",
            )
            inputs.append(val)

    # Convert user inputs to model format
    processed_inputs = inputs.copy()
    processed_inputs[0] = processed_inputs[0] / 10000  # back to tens of thousands

    if st.button("Predict House Value"):
        payload = {"features": processed_inputs}
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            prediction = data['prediction'] * 100_000
            st.success(f"Predicted Median House Value: ${prediction:,.0f}")
        except Exception as e:
            st.error(f"Error contacting API: {e}")

with tabs[1]:
    st.subheader("Feature Distributions vs Your Input")

    selected_feature = st.selectbox("Select a feature to visualize:", list(ui_to_feature.keys()))
    feature = ui_to_feature[selected_feature]
    values = X[feature]

    plot_values = values * 10000 if feature == "MedInc" else values

    discrete_features = ["AveRooms", "AveBedrms", "HouseAge"]
    
    if feature in discrete_features:
        bins = int(np.ceil(plot_values.max() - plot_values.min() + 1))
    else:
        bins = 50

    if feature not in discrete_features:
        lower = np.percentile(plot_values, 0.5)
        upper = np.percentile(plot_values, 99.5)
    else:
        lower = plot_values.min() - 0.5
        upper = plot_values.max() + 0.5

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(plot_values, bins=bins, ax=ax, kde=False, color="#1f77b4", alpha=0.7)

    user_val = inputs[list(default_inputs.keys()).index(selected_feature)]
    ax.axvline(user_val, color="red", linestyle="--", label="Your input")

    ax.set_xlim(lower, upper)
    ax.set_title(f"{selected_feature} distribution")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


