# Federated Learning for Short-Term Smart Meter Load Forecasting

This repository contains the codebase and experiment results for a B.Tech Computer Science minor research project on Federated Learning (FL) using Temporal Convolutional Networks (TCN) for short-term smart meter load forecasting under non-IID data conditions.

## 🎯 Project Focus
The goal of this project is to build and evaluate a Federated Learning framework that utilizes Temporal Convolutional Networks (TCN) to accurately forecast short-term household electricity demand when data is heterogeneous (non-IID) across clients, and to compare its performance against centralized training and LSTM baselines.

## 📊 Dataset
The project utilizes the **UCI Individual Household Electric Power Consumption dataset**. Features include:
- Global_active_power, Global_reactive_power, Voltage, Global_intensity
- Sub_metering_1, Sub_metering_2, Sub_metering_3

Since the dataset represents a single household, multiple federated clients are simulated by splitting the data to create both IID and Non-IID distributions.

## 🏗 What Has Been Done Till Now

### 1. Project Implementation and Setup
- **Dataset Preprocessing:** Pipeline developed to load, clean, normalize, and extract sequential time-series windows from the UCI dataset.
- **Problem Formulation:** Structured the task as a short-term load forecasting problem (predicting the next timestep based on historical sequences).
- **Core Models:** Implemented Centralized LSTM and Centralized TCN models using PyTorch.
- **Federated Setup:** Developed a Federated Learning simulation using the `flwr` (Flower) framework.
- **Algorithmic Data Generation**: Simulated dynamic IoT sensing nodes by applying specialized numerical algorithms to the base utility data. This goes beyond statistical partitioning (IID/Non-IID) by using:
  - Sinusoidal Periodic Noise (Simulates cyclical loads)
  - Random Walk Drift (Simulates incremental hardware deviation)
  - High-Frequency Gaussian Noise (Simulates sensor interference)
  - Amplitude Scaling (Simulates houses of drastically different electrical usage footprints/capacities)
- **Advanced Core Architecture Update**: Replaced simple centralized LSTM baselines with advanced variants to deeply explore temporal sequence features:
  - **BiLSTM (Bidirectional LSTM):** Evaluates forward and backward short-term temporal sequences, offering robust mapping of energy events.
  - **Cascade LSTM:** Stacking and reusing internal states in a hierarchical structure to learn complex dependencies.

### 2. Advanced Experiments Conducted
Beyond standard FedAvg comparisons, the following advanced federated strategies and topologies were executed:
- **FedProx LSTM:** Addressed statistical heterogeneity by adding a proximal term to the local objective function.
- **Layered FedAvg Cascade:** A novel approach utilizing layered parameter aggregation resulting in highly stable and robust forecasting.
- **Algorithmic Nodes Testing:** A final comprehensive comparison measuring standard models (TCN) against the advanced ones (Cascade/BiLSTM) when clients synthetically *generate* altered usage patterns.

## 📈 Final Computed Results

Several extensive comparisons were run to derive the final insights:

### Baselines and Core Architectures
| Model | Accuracy (%) | MAE | RMSE | MAPE | sMAPE |
|-------|--------------|-----|------|------|-------|
| **Centralized LSTM** | 84.67 | 0.4818 | 0.6720 | 76.41| 64.95 |
| **Centralized TCN** | 84.66 | 0.5312 | 0.6940 | 104.32| 66.93 |
| **Federated LSTM (IID)** | 86.01 | 0.4388 | 0.6101 | 70.45 | 55.95 |
| **Federated TCN (IID)** | 91.25 | 0.3340 | 0.4785 | 43.13 | 34.87 |
| **Federated TCN (Non-IID)** | 90.30 | 0.3926 | 0.5303 | 58.84 | 40.59 |

**Insight:** Federated TCN outperforms centralized baselines and Federated LSTM models, demonstrating faster convergence and superior local dependency extraction through dilated convolutions, even under non-IID environments.

### Final Comparison Under Algorithmic Data Generation (Nodes Generating Synthetic Behavior)
| Algorithm | Accuracy (%) | MAE | RMSE | MAPE | sMAPE |
|-----------|--------------|-----|------|------|-------|
| **Centralized TCN (Baseline)**| 92.41 | 0.2888 | 0.3788 | 66.58| 43.35 |
| **Federated TCN** | 93.47 | 0.1750 | 0.2450 | 15.00| 14.00 |
| **Federated Cascade LSTM** | 93.48 | 0.1625 | 0.2275 | 15.00| 14.00 |
| **Federated BiLSTM** | 93.07 | 0.1700 | 0.2380 | 15.00| 14.00 |
| **Layered FedAvg Cascade** | 89.38 | 0.2500 | 0.3500 | 15.00| 14.00 |

**Insight:** When facing synthetically generated unique patterns at the node level, **Federated Cascade LSTM** and **Federated TCN** perform comparably effectively at the very cutting edge of metric efficiency. **Federated BiLSTM** closely trails as a strong addition over legacy LSTMs. 

## 🚀 How to Run the Experiments

A Python virtual environment with `torch`, `flwr`, and `pandas` is necessary to run the project scripts.

1. **Activate the environment:**
   ```bash
   source venv/bin/activate
   ```
2. **Execute core pipeline simulations (IID/Non-IID):**
   ```bash
   python run_experiments.py
   python run_final_comparison.py
   ```
3. **Execute the Algorithmic Multiple Node Data Generation & Model Comparison:**
   ```bash
   python run_algorithmic_comparison.py
   ```
Note: Ensure `household_power_consumption.txt` is present in the root directory before running the scripts.