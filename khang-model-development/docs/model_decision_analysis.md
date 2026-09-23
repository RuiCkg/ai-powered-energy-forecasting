# Architectural Decision Analysis: LSTM vs. GRU vs. TCN

**Author:** Khang Huynh (AI/ML Model Lead)  
**Project:** AI-Powered Energy Forecasting Platform  

---

## 1. Executive Summary
This document outlines the theoretical and empirical evaluation comparing **LSTM (Long Short-Term Memory)** against alternative sequence architectures, specifically **GRU (Gated Recurrent Unit)** and **TCN (Temporal Convolutional Network)**, justifying why LSTM was retained as the advanced sequence model for the platform[cite: 3].

## 2. Comparative Matrix

| Evaluation Criteria | Stacked LSTM (Selected)[cite: 3] | Gated Recurrent Unit (GRU)[cite: 3] | Temporal Convolutional Network (TCN)[cite: 3] |
| :--- | :--- | :--- | :--- |
| **Gating Architecture** | 3 Gates (Input, Forget, Output) + Dedicated Cell State $C_t$[cite: 3]. | 2 Gates (Reset, Update); no separate cell state[cite: 3]. | Dilated 1D Causal Convolutions + Residual Blocks[cite: 3]. |
| **Long-Term Memory Retention** | Superior long-horizon error propagation over 168-hour seasonalities[cite: 3]. | Slight error degradation over extended sequence windows[cite: 3]. | High receptive field, but requires deep stack for long lags[cite: 3]. |
| **Training Execution Speed** | Moderate (~18s per 50 epochs on GPU)[cite: 3, 5]. | 25% faster per epoch due to reduced parameter count[cite: 3]. | Fast parallel training across time steps[cite: 3]. |
| **Parameter Tuning Risk** | Stable convergence with standard Adam optimizer[cite: 3]. | Sensitive to learning rate spikes on noisy load spikes[cite: 3]. | Complex hyperparameter search (kernel size, dilation factor)[cite: 3]. |

## 3. Decision Justification
1. **Cell State Mechanics:** Energy demand features distinct daily (24h) and weekly (168h) temporal patterns[cite: 3]. LSTM's separate cell state ($C_t$) acts as an unattenuated memory line, preventing historical temporal gradients from vanishing across long lag sequences[cite: 3].
2. **Complementary Paradigm to XGBoost:** XGBoost models static tabular interactions, whereas LSTM provides a true recurrent state space model[cite: 3].
3. **Resource Risk Mitigation:** To manage Google Colab free GPU limits (Risk R1/R5)[cite: 3, 5], PyTorch `EarlyStopping` was introduced, capping training execution under 20 seconds and satisfying NFR-1[cite: 3, 4].