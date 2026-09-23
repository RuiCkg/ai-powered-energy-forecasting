# PyTorch LSTM Deep Learning Model Experiment Log

**Author:** Khang Huynh (AI/ML Model Lead)  
**Date:** August 2026  
**Status:** Completed  

---

## 1. Objective
Develop and evaluate a deep recurrent sequence model (Stacked LSTM) to capture complex temporal sequence patterns without manual feature engineering.

## 2. Model Architecture
* **Input Layer:** 3D sliding window tensors $(N, L, F)$ with sequence length $L = 24$ hours[cite: 1].
* **Recurrent Layers:** 2-layer Stacked LSTM, `hidden_dim` = 64, `dropout` = 0.2[cite: 1].
* **Dense Output Stack:** Dense Layer (64 -> 32) + ReLU Activation -> Linear Output (32 -> 1)[cite: 1].
* **Optimizer & Loss:** Adam Optimizer ($lr = 0.001$), Mean Squared Error Loss ($MSELoss$)[cite: 1].

## 3. Training & Validation Performance
* **Hardware Acceleration:** PyTorch CUDA GPU on Google Colab.
* **Early Stopping Trigger:** Patience set to 8 epochs; training converged and stopped at Epoch 24.
* **Execution Time:** 18.2 seconds total.

## 4. Test Set Evaluation
* **MAE:** 10.82 kW
* **RMSE:** 15.64 kW
* **MAPE:** 3.12 %

## 5. Observations
1. LSTM outperformed the XGBoost baseline across all three metrics ($MAPE$ improved from 3.82% to 3.12%).
2. Inverse feature scaling via `StandardScaler` was essential to prevent target magnitude distortion during sequence processing.