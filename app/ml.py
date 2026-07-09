import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def build_sequences(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_model(series: np.ndarray, window: int, model_name: str):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    X, y = build_sequences(scaled, window)
    split = int(len(X) * 0.85)

    if model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )

    model.fit(X[:split], y[:split])
    return model, scaler


def predict_future(model, scaler, last_window: np.ndarray, n_days: int) -> np.ndarray:
    scaled = scaler.transform(last_window.reshape(-1, 1)).flatten()
    current = list(scaled)
    w = len(last_window)
    preds = []
    for _ in range(n_days):
        x = np.array(current[-w:]).reshape(1, -1)
        p = model.predict(x)[0]
        preds.append(p)
        current.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


def fit_metrics(model, scaler, series: np.ndarray, window: int):
    scaled_full = scaler.transform(series.reshape(-1, 1)).flatten()
    X_all, _ = build_sequences(scaled_full, window)
    y_pred = scaler.inverse_transform(model.predict(X_all).reshape(-1, 1)).flatten()
    y_true = series[window:]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"mae": float(mae), "rmse": rmse, "mape": mape, "y_pred": y_pred}
