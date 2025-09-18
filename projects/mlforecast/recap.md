# MLForecast + XGBoost Forecasting

## 📚 Official Description (MLForecast)

From MLForecast’s documentation: 
[1]: https://nixtlaverse.nixtla.io/mlforecast/index.html

> MLForecast is a framework for time series forecasting using machine learning models, designed to **scale to massive amounts of data** using remote clusters. ([nixtlaverse.nixtla.io][1])

Key features include:

* Very fast, efficient feature‐engineering for time series. ([nixtlaverse.nixtla.io][1])
* Out‐of‐the‐box compatibility with pandas, polars, spark, dask, ray. ([nixtlaverse.nixtla.io][1])
* Support for exogenous variables and static covariates. ([nixtlaverse.nixtla.io][1])
* Familiar sklearn‐style `.fit(...)` / `.predict(...)` interface. ([nixtlaverse.nixtla.io][1])

---

## ⚙️ Model Overview

This pipeline uses **MLForecast** with **XGBoost** to forecast future values for multiple time series. Below are its main components:

* **Feature Engineering**:

  * Lag features (e.g. lags = \[1,2,3,6,12])
  * Lag transforms like **Rolling Mean**, **Expanding Mean**
  * Seasonal differencing (e.g. differencing lag 12)

* **Model**:

  * `XGBRegressor` from XGBoost, with typical hyperparameters (n\_estimators, max\_depth, etc.)

* **Evaluation**:

  * Forecast horizon usually short (e.g. 3 periods)
  * Metrics: WMAPE, MAPE

---

## 🔧 Dependencies

* `pandas` / `numpy` — data handling
* `mlforecast` — core forecasting framework (lag & target transforms, etc.)
* `xgboost` — the ML model (You can change any model that you wanna test with MLForecast, but the "boosting" models are especially robust for the latest version of MLForecast)
* `scikit‐learn` — for evaluation metrics and compatibility
* `warnings` to suppress non‐critical logs

---

## 📉 Current Status & Limitations

What’s working / done:

* Data loading & format conversion into MLForecast format
* Building the model (lags, transforms, etc.)
* Training & short‐horizon prediction works
* Basic evaluation when ground truth in test data

What’s not working / open issues:

* Some file paths are hard‐coded to your personal directories → must change for portability
* Handling missing values / nulls needs more robust logic in some datasets
* Hyperparameter tuning not yet done (learning rate, lag selection, etc.)
* Longer horizon forecasts less stable / less accurate (no strong validation yet)
* Cross‐validation function is present but sometimes fails or gives sparse output

---

## 🚀 Next Steps / Recommendations

To help whoever takes over:

1. Replace hard‐coded paths with relative or configurable paths (e.g. via config file or command line args).
2. Add more robust missing / null treatment, e.g. imputation, or dynamic handling per series.
3. Run hyperparameter search (grid or random) over XGBoost settings & lag possibilities.
4. Expand the evaluation: more metrics (MAE, RMSE, etc.), perhaps backtesting / multiple windows.
5. Add unit tests or small examples so the pipeline can be validated easily.
6. Document how to deploy or run with new data (data format expectations, column types, frequency assumptions).
