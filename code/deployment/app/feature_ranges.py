from sklearn.datasets import fetch_california_housing
import numpy as np

# Load dataset once and compute ranges
data = fetch_california_housing(as_frame=True)
X = data.frame

feature_stats = {
    "MedInc": {"min": float(X["MedInc"].min()), "max": float(X["MedInc"].max())},
    "HouseAge": {"min": float(X["HouseAge"].min()), "max": float(X["HouseAge"].max())},
    "AveRooms": {"min": float(X["AveRooms"].min()), "max": float(X["AveRooms"].max())},
    "AveBedrms": {"min": float(X["AveBedrms"].min()), "max": float(X["AveBedrms"].max())},
    "Population": {"min": float(X["Population"].min()), "max": float(X["Population"].max())},
    "AveOccup": {"min": float(X["AveOccup"].min()), "max": float(X["AveOccup"].max())},
    "Latitude": {"min": float(X["Latitude"].min()), "max": float(X["Latitude"].max())},
    "Longitude": {"min": float(X["Longitude"].min()), "max": float(X["Longitude"].max())},
}
