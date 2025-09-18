# File: mlforecast_xgb.py
# Description: Time series forecasting pipeline using MLforecast with XGBoost.
#              Includes preprocessing, training, prediction, evaluation, and optional cross-validation.

import pandas as pd
import numpy as np
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(train_path, test_path):
    """Load train/test data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain columns:", train_df.columns.tolist())
    print("\nTrain head:\n", train_df.head())

    return train_df, test_df


def prepare_mlforecast_format(df, date_col='feature_9', target_col='feature_10',
                              group_cols=['feature_3', 'feature_7']):
    """Convert raw data into MLforecast format."""

    # Build unique_id from group columns
    if len(group_cols) > 1:
        df['unique_id'] = df[group_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    else:
        df['unique_id'] = df[group_cols[0]].astype(str)

    ml_df = df[['unique_id', date_col, target_col]].copy()
    ml_df.rename(columns={date_col: 'ds', target_col: 'y'}, inplace=True)

    ml_df['ds'] = pd.to_datetime(ml_df['ds'])
    ml_df = ml_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    ml_df = ml_df.dropna()

    print(f"\nFormatted shape: {ml_df.shape}")
    print(f"Number of series: {ml_df['unique_id'].nunique()}")
    print(f"Date range: {ml_df['ds'].min()} → {ml_df['ds'].max()}")

    return ml_df


def create_mlforecast_model():
    """Create MLforecast model with XGBoost."""

    lags = [1, 2, 3, 6, 12]
    lag_transforms = {
        1: [ExpandingMean()],
        3: [RollingMean(window_size=3)],
        6: [RollingMean(window_size=6)],
        12: [RollingMean(window_size=12)],
    }

    xgb_model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    mlf = MLForecast(
        models=[xgb_model],
        freq='M',  # monthly frequency
        lags=lags,
        lag_transforms=lag_transforms,
        target_transforms=[Differences([12])],  # seasonal differencing
        num_threads=1
    )
    return mlf


def train_and_predict(mlf, train_df, horizon=3):
    """Train model and forecast future values."""
    print("\nTraining model...")
    mlf.fit(train_df)
    print("Training completed.")

    print(f"\nForecasting next {horizon} steps...")
    predictions = mlf.predict(h=horizon)

    print("Prediction completed.")
    print(f"Predictions shape: {predictions.shape}")
    print("\nSample predictions:\n", predictions.head(10))
    return predictions


def evaluate_predictions(predictions, test_df, target_col='feature_10'):
    """Evaluate predictions if test data has ground truth."""
    if target_col in test_df.columns:
        test_formatted = prepare_mlforecast_format(test_df)
        merged = predictions.merge(test_formatted, on=['unique_id', 'ds'], how='inner')

        if not merged.empty:
            wmape = np.sum(np.abs(merged['y'] - merged['XGBRegressor'])) / np.sum(np.abs(merged['y'])) * 100
            mape = mean_absolute_percentage_error(merged['y'], merged['XGBRegressor']) * 100
            print(f"\nEvaluation results:\nWMAPE: {wmape:.4f}%\nMAPE: {mape:.4f}%")
            return wmape, mape
    return None, None


def cross_validation_analysis(mlf, train_df, n_windows=3, h=3):
    """Perform rolling-window cross-validation."""
    print(f"\nCross-validation (windows={n_windows}, horizon={h})...")
    try:
        cv_results = mlf.cross_validation(df=train_df, n_windows=n_windows, h=h, step_size=1)
        print("Cross-validation completed.")
        print(f"CV results shape: {cv_results.shape}")

        wmape = np.sum(np.abs(cv_results['y'] - cv_results['XGBRegressor'])) / np.sum(np.abs(cv_results['y'])) * 100
        mape = mean_absolute_percentage_error(cv_results['y'], cv_results['XGBRegressor']) * 100

        print(f"\nCV Metrics:\nCV WMAPE: {wmape:.4f}%\nCV MAPE: {mape:.4f}%")
        return cv_results, wmape, mape
    except Exception as e:
        print(f"Cross-validation failed: {str(e)}")
        return None, None, None


def save_predictions(predictions, output_path):
    """Save predictions to CSV."""
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


def main():
    """Main entry point."""

    train_path = "data/client_train.csv"
    test_path = "data/client_test.csv"
    output_path = "mlforecast_predictions.csv"

    print("=== MLforecast Time Series Forecasting ===")

    try:
        train_df, test_df = load_and_prepare_data(train_path, test_path)
        train_formatted = prepare_mlforecast_format(train_df)

        mlf = create_mlforecast_model()

        # Optional: run CV
        # cv_results, cv_wmape, cv_mape = cross_validation_analysis(mlf, train_formatted, n_windows=2, h=3)

        predictions = train_and_predict(mlf, train_formatted, horizon=3)
        wmape_score, mape_score = evaluate_predictions(predictions, test_df)

        save_predictions(predictions, output_path)

        print("\n=== Forecasting Completed ===")
        return predictions, mlf

    except Exception as e:
        print(f"Execution error: {str(e)}")
        raise


if __name__ == "__main__":
    preds, model = main()
    print("\n=== Extra Info ===")
    print("Available objects:")
    print("- preds: prediction DataFrame")
    print("- model: trained MLforecast model")
    print("- Use model.predict(h=6) for longer horizons.")
