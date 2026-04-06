# Research Progress Report: Federated Learning for Short-Term Smart Meter Load Forecasting

## 1. Introduction and Motivation (Why This Topic Was Chosen)
Predicting short-term electricity demand is an essential component of stabilizing and managing modern smart grids. However, obtaining high-quality data to train these predictive models presents a significant hurdle. Centralized data collection of smart meter readings raises massive privacy concerns for consumers. Furthermore, every household has a drastically different electrical usage footprint, leading to data that is highly heterogeneous or "non-IID" (Non-Independent and Identically Distributed). 

This project was chosen to bridge this gap by establishing a **Federated Learning (FL)** framework. By utilizing FL, we can train robust, highly accurate temporal machine learning models to forecast power loads across diverse households without ever requiring sensitive user data to leave the local edge devices (smart meters).

## 2. Problem Statement and Backlogs (Challenges Faced)
During the initial phases of the project, several backlogs and conceptual challenges were identified in existing approaches:
* **The Generalization Gap:** Standard centralized models (like basic LSTMs) struggle to generalize well when tested across diverse households. A single global model trained centrally fails to capture the unique, localized usage topologies of individual homes.
* **Real-World IoT Simulation:** Simply splitting a dataset statistically (into IID or non-IID partitions) does not accurately reflect real-world physics or IoT sensor behaviors. Smart meters are prone to cyclical loads, hardware degradation, and interference.
* **Privacy Risks:** Legacy systems require all continuous time-series data to be aggregated in a central data lake, which is a major security vulnerability and violates user privacy principles.

## 3. Improvements and Proposed Methodology (How We Did It)
To overcome the limitations of centralized and standard federated approaches, we engineered several key improvements in our methodology:

### A. Algorithmic Data Generation
Instead of relying on simple statistical partitioning, we dynamically simulated realistic IoT sensing nodes using advanced numerical algorithms on the UCI Individual Household Electric Power Consumption dataset. This included injecting realistic anomalies:
* **Sinusoidal Periodic Noise:** To simulate cyclical loads and daily/weekly usage patterns.
* **Random Walk Drift:** To simulate incremental hardware deviations and degradation over time.
* **High-Frequency Gaussian Noise:** To replicate sensor interference and communication noise.
* **Amplitude Scaling:** To simulate houses with drastically different electrical usage footprints and capacities.

### B. Federated Architecture Setup
We transitioned from a centralized paradigm to a decentralized simulation using the Python `flwr` (Flower) framework. This allowed us to emulate multiple independent edge nodes training their own local models.

### C. Advanced Aggregation Strategies
Standard federated averaging (FedAvg) was insufficient for our generated non-IID data. We explored robust strategies like **FedProx** (to address statistical heterogeneity by adding a proximal term to the local objective) and developed a novel **Layered FedAvg Cascade** approach for highly stable parameter aggregation.

## 4. Machine Learning Architectures Used
To deeply explore temporal sequence features, we transitioned from basic baselines to advanced neural network architectures:

**Centralized Baselines:**
* **Centralized LSTM:** Standard Long Short-Term Memory network.
* **Centralized TCN:** Temporal Convolutional Network utilizing dilated causal convolutions.

**Advanced Federated Models:**
* **Federated TCN:** Extends the TCN into a decentralized environment, extracting local dependencies rapidly.
* **Federated BiLSTM (Bidirectional LSTM):** Evaluates forward and backward short-term temporal sequences, offering robust mapping of energy events.
* **Federated Cascade LSTM:** A state-of-the-art architecture that stacks and reuses internal states hierarchically to learn complex, long-term dependencies.

## 5. System Workflow (How the Machine Learning Operates)
The operational workflow of the ML framework dynamically functions as follows:
1. **Data Preprocessing:** The system frames the task as a sequence prediction problem. It dynamically extracts sequential time-series windows from historical features. During our feature set experiments, we found that combining Intensity, Voltage, and Sub-metering yields strong metrics.
2. **Local Edge Training:** At the client node (the simulated smart home), the ML model extracts temporal features locally. 
3. **Parameter Transmission:** Instead of sending raw power usage, the local client transmits only the computed model weights (parameters) to the central server.
4. **Global Aggregation:** The main server aggregates these varied weights using our advanced federated algorithms to establish a smarter, generalized global model, which is then broadcasted back to the edge nodes for the next round of training.

## 6. Final Evaluated Results (Based on Latest Core Output)

Comprehensive experimental comparisons were conducted, split into Core Setup Evaluations and Advanced Algorithmic Node Simulations.

### A. Comprehensive Setup & Baselines (Full Feature Set)
Under a traditional statistical split, standard and robust Federated models greatly out-paced Centralized counterparts:
| Model / Algorithm | Accuracy (%) | MAE | RMSE | MAPE (%) |
|-------------------|--------------|-----|------|----------|
| **Centralized LSTM** | ~82.32% | 0.562 | 0.703 | 81.23% |
| **Centralized TCN** | ~82.54% | 0.510 | 0.668 | 62.34% |
| **Federated LSTM (IID split)** | ~90.80% | 0.310 | 0.427 | 51.08% |
| **Federated TCN (Non-IID split)**| ~90.37% | 0.328 | 0.447 | 52.91% |

### B. Algorithmic Data Generation Limits (Simulating Highly Unique IoT Nodes)
When exposed to dynamically generated cyclic noises, amplitude scales, and drifts (representing severe non-IID conditions):

| Model / Algorithm | Accuracy (%) | MAE | RMSE | sMAPE (%) |
|-------------------|--------------|-----|------|----------|
| **Centralized TCN** | 91.50% | 0.288 | 0.376 | 43.40% |
| **Federated TCN** | 91.74% | 0.496 | 0.570 | 65.59% |
| **Federated BiLSTM** | 93.30% | 0.263 | 0.345 | 41.53% |
| **Federated Cascade LSTM** | **94.39%** | **0.230**| **0.318**| **37.88%** |

**Key Insight:** Under the most chaotic, realistic simulation conditions (`Algorithmic Nodes`), the **Federated Cascade LSTM** drastically outperformed all counterparts, elevating accuracy to a peak **94.39%** while sharply minimizing error indices (MAE dropped to 0.230, and symmetric MAPE dropped efficiently to 37.88%).

## 7. Conclusion
The research definitively proves that deploying advanced temporal architectures—most notably the **Cascade LSTMs and BiLSTMs**—within a Federated Learning framework is exceptionally effective for smart energy grids. It circumvents the severe privacy risks associated with centralized data lakes, whilst overcoming extreme hardware and statistical heterogeneity across power networks to deliver a 94%+ predictive accuracy.
