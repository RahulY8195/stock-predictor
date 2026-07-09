import numpy as np

from app.ml import build_sequences, fit_metrics, predict_future, train_model


def test_build_sequences_shapes():
    series = np.arange(50, dtype=float)
    window = 10
    X, y = build_sequences(series, window)

    assert X.shape == (40, window)
    assert y.shape == (40,)
    np.testing.assert_allclose(X[0], series[:window])
    assert y[0] == series[window]


def test_train_model_and_predict_future_gradient_boosting(synthetic_price_df):
    series = synthetic_price_df["Close"].values.astype(float)
    window = 30

    model, scaler = train_model(series, window, "Gradient Boosting")
    forecast = predict_future(model, scaler, series[-window:], n_days=10)

    assert forecast.shape == (10,)
    assert np.all(np.isfinite(forecast))
    assert forecast.min() > 50
    assert forecast.max() < 250


def test_train_model_and_predict_future_mlp(synthetic_price_df):
    series = synthetic_price_df["Close"].values.astype(float)
    window = 30

    model, scaler = train_model(series, window, "MLP Neural Network")
    forecast = predict_future(model, scaler, series[-window:], n_days=10)

    assert forecast.shape == (10,)
    assert np.all(np.isfinite(forecast))


def test_fit_metrics_reports_low_error_on_learnable_trend(synthetic_price_df):
    series = synthetic_price_df["Close"].values.astype(float)
    window = 30

    model, scaler = train_model(series, window, "Gradient Boosting")
    metrics = fit_metrics(model, scaler, series, window)

    assert set(metrics.keys()) == {"mae", "rmse", "mape", "y_pred"}
    assert metrics["mae"] >= 0
    assert metrics["mape"] < 25
