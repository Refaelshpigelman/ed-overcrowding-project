import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocessing import load_data, preprocess_data, create_hourly_target

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "arrivals_lag_1",
    "arrivals_lag_2",
    "arrivals_roll_3",
    "arrivals_roll_6",
]

TARGET = "arrivals"

def train_and_evaluate(csv_path: str):
    df = load_data(csv_path)
    clean_df = preprocess_data(df)
    hourly_df = create_hourly_target(clean_df)

    split_idx = int(len(hourly_df) * 0.8)
    train_df = hourly_df.iloc[:split_idx].copy()
    test_df = hourly_df.iloc[split_idx:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R2": r2_score(y_test, preds),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    prediction_df = test_df[["timestamp", TARGET]].copy()
    prediction_df["prediction"] = preds
    return model, results, prediction_df

if __name__ == "__main__":
    model, results, prediction_df = train_and_evaluate("data/ED_full_data_2.csv")
    print(results)
    print(prediction_df.head())
