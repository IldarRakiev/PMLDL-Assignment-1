from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs("../../models", exist_ok=True)

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model: Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")

    model_path = os.path.join(os.path.dirname(__file__), "../../models/california_gb.joblib")
    joblib.dump(
        {"model": model, "feature_names": list(data.feature_names), "target_name": "MedianHouseValue"},
        model_path
    )
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
