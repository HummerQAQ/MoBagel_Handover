# Zero-Inflated State Space Model (ZISSM)

## 📌 Model Overview

The **Zero-Inflated State Space Model (ZISSM)** is designed for **intermittent demand forecasting** where the target time series contains many zeros.
It combines:

* **State Space Model (SSM)** for latent temporal dynamics
* **Zero-Inflated Poisson (ZIP)** likelihood to handle zero-heavy distributions

This allows the model to capture both **temporal dependency** and **zero-inflation** in sales data, making it suitable for sparse retail demand forecasting.

---

## ⚙️ Key Features

* **Zero-Inflation Handling**: Explicitly models excess zeros with a logistic gate.
* **State Dynamics**: Hidden state evolves with an AR(1)-like process.
* **Poisson Count Modeling**: Observations follow a Zero-Inflated Poisson distribution.
* **Bayesian Inference**: Uses **Stochastic Variational Inference (SVI)** in Pyro.
* **GPU Support**: Automatically detects and uses CUDA for faster training.

---

## 🛠️ Dependencies

The following libraries are required:

* **[PyTorch](https://pytorch.org/)** → tensor operations and GPU acceleration
* **[Pyro](https://pyro.ai/)** → probabilistic programming and variational inference
* **Pandas / NumPy** → data handling
* **Matplotlib** → visualization

---

## 🧩 Model Principle

1. **Latent State Transition**

   $$
   z_t \sim \mathcal{N}(\alpha \cdot z_{t-1}, \sigma^2)
   $$

   where $\alpha$ controls temporal correlation.

2. **Observation Distribution**

   $$
   y_t \sim \text{ZeroInflatedPoisson}(\pi_t, \lambda_t)
   $$

   with

   $$
   \pi_t = \sigma(\text{logit}_\pi + z_t), \quad \lambda_t = \text{base\_rate} \cdot e^{z_t}
   $$

   where $\pi_t$ is the probability of extra zeros.

3. **Inference**
   Variational inference approximates the posterior of latent states $z_t$.

---

## 📉 Current Development Status

* ✅ Data preprocessing and train/test split implemented
* ✅ GPU support enabled
* ✅ Training loop (SVI) and prediction pipeline implemented
* ⚠️ **Issue**: Model training does not converge properly

  * Likely due to an **incomplete/unstable model definition** (e.g., latent state initialization or observation plate setup).
  * Future work: refine the model structure (e.g., hierarchical priors, better latent dynamics, constraints).

---

## 🚀 Next Steps

1. Refine model definition for better stability:

   * Consider constraining AR coefficient `alpha` within (-1, 1).
   * Add hierarchical priors to stabilize latent states.
   * Explore alternative likelihoods (Negative Binomial).
2. Perform hyperparameter tuning on learning rate and prior scales.
3. Add evaluation metrics (MAE, RMSE, MASE) for forecasting comparison.

---