# File: dlinear_benchmark.py
# Description: Benchmark script for multiple datasets using DLinear model from NeuralForecast.
#              Handles preprocessing, model training, prediction, and evaluation.

import pandas as pd
import numpy as np
import time
import os
import warnings

# Limit CPU threads for reproducibility
import torch
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from neuralforecast.losses.pytorch import MAE

# Settings
warnings.filterwarnings('ignore')

# Benchmark configurations (without exogenous variables)
configs = [
    {
        'name': 'M5_Forecasting-Accuracy',
        'target_col': 'sales', 'datetime_col': 'date',
        'time_groups': ['state_id', 'store_id'], 'horizon': 28, 'freq': 'D',
    },
    {
        'name': 'Rossmann-Sales',
        'target_col': 'Sales', 'datetime_col': 'date',
        'time_groups': ['Store'], 'horizon': 30, 'freq': 'D',
    },
    {
        'name': 'Restaurant_Visitor_Forecasting',
        'target_col': 'visitors', 'datetime_col': 'date',
        'time_groups': ['air_store_id'], 'horizon': 14, 'freq': 'D',
    },
    # ... add more configs as needed
]

# Input/Output paths (update as needed)
base_path = "data/benchmark"
output_path = "results/dlinear"
os.makedirs(output_path, exist_ok=True)
results_log_file = os.path.join(output_path, 'dlinear_benchmark_results.csv')


# Weighted Mean Absolute Percentage Error
def wmape(y_true, y_pred):
    abs_y_true = np.abs(y_true)
    if np.sum(abs_y_true) < 1e-9:
        return np.mean(np.abs(y_true - y_pred))
    return np.sum(np.abs(y_true - y_pred)) / np.sum(abs_y_true)


def main():
    all_results = []

    for config in configs:
        clean_name = config['name'].replace('_train', '').replace('_test', '')

        try:
            start_time = time.time()
            print(f"--- Running Benchmark: {clean_name} ---")

            # Load train/test data
            train_path = os.path.join(base_path, clean_name, f"{clean_name}_train.csv")
            test_path = os.path.join(base_path, clean_name, f"{clean_name}_test.csv")

            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)

            # Drop rows with missing group keys
            train_df.dropna(subset=config['time_groups'], inplace=True)
            test_df.dropna(subset=config['time_groups'], inplace=True)

            # Merge train + test (test with target removed)
            full_df = pd.concat(
                [train_df, test_df.drop(columns=[config['target_col']], errors='ignore')],
                ignore_index=True
            )

            # Rename required columns
            full_df.rename(columns={config['datetime_col']: 'ds', config['target_col']: 'y'}, inplace=True)
            full_df['ds'] = pd.to_datetime(full_df['ds'], errors='coerce')
            full_df.dropna(subset=['ds'], inplace=True)
            full_df['y'] = pd.to_numeric(full_df['y'], errors='coerce')

            # Create unique_id
            full_df['unique_id'] = full_df[config['time_groups']].astype(str).agg('_'.join, axis=1)

            # Keep only required columns
            df_min = full_df[['unique_id', 'ds', 'y']].copy()

            # Determine train end date
            train_end_date = pd.to_datetime(train_df[config['datetime_col']].max())

            # Use last 100 observations per series for training
            Y_train_df = (
                df_min[df_min['ds'] <= train_end_date]
                .groupby('unique_id').tail(100).copy()
            )
            Y_train_df['y'] = Y_train_df['y'].fillna(0)
            if Y_train_df.empty:
                raise ValueError("Training data is empty after preprocessing.")

            # Model training & prediction
            h = config['horizon']
            input_size = min(99, 2 * h) if h > 0 else 32
            VAL_SIZE = max(h, 3)

            model = DLinear(
                """
                Early stopping is enabled, it will strongly impact the performance of the model from NeuralForecast.
                If your dataset is very small or noisy, early stopping may trigger before the model fully converges. 
                Consider adjusting 'early_stop_patience_steps' or disabling early stopping if needed for your use case.
                """,
                h=h,
                input_size=input_size,
                loss=MAE(),
                max_steps=5000,
                val_check_steps=20,
                early_stop_patience_steps=50,
                scaler_type='standard',
                random_seed=777,
                start_padding_enabled=True,
                accelerator='cpu',
                dataloader_kwargs={'num_workers': 0}  # Avoid multiprocessing issues on Windows
            )

            nf = NeuralForecast(models=[model], freq=config['freq'])
            nf.fit(df=Y_train_df, val_size=VAL_SIZE)

            predictions = nf.predict()  # returns DataFrame with 'DLinear' column

            # Evaluation
            test_ground_truth = test_df.rename(columns={config['datetime_col']: 'ds', config['target_col']: 'y'})
            test_ground_truth['ds'] = pd.to_datetime(test_ground_truth['ds'])
            test_ground_truth['unique_id'] = test_ground_truth[config['time_groups']].astype(str).agg('_'.join, axis=1)

            results_df = pd.merge(
                test_ground_truth[['unique_id', 'ds', 'y']],
                predictions,
                on=['unique_id', 'ds'],
                how='inner'
            )

            y_true, y_pred = results_df['y'].values, results_df['DLinear'].values
            score = wmape(y_true, y_pred)
            score_type = 'wmape' if np.sum(np.abs(y_true)) > 1e-9 else 'mae'

            execution_time = time.time() - start_time
            result = {'benchmark': clean_name, f'score ({score_type})': score, 'execution_time_sec': execution_time}
            all_results.append(result)
            print(f"Result: {result}\n")

        except Exception as e:
            import traceback
            print(f"!!!!!! FAILED on Benchmark: {clean_name} !!!!!!\nError: {e}")
            traceback.print_exc()
            all_results.append({'benchmark': clean_name, 'wmape': 'FAILED', 'execution_time_sec': -1})

    # Save results
    results_df_final = pd.DataFrame(all_results)
    results_df_final.to_csv(results_log_file, index=False)
    print(f"--- All Benchmarks Finished ---\nResults saved to: {results_log_file}\n\nFinal Summary:\n{results_df_final}")


if __name__ == "__main__":
    main()
