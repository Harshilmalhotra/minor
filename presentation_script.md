# Project Guide Presentation Script: Federated TCN for Load Forecasting

## 1. Introduction & Objective

"Hello sir/ma'am. My minor project is titled **'Federated Learning with Temporal Convolutional Networks for Short-Term Load Forecasting under Non-IID Conditions.'**

The core problem I am solving is **Smart Grid Privacy**. Currently, to predict electricity demand, utility companies collect raw, high-resolution power consumption data from every home into a central server. This is a massive privacy violation, as power data reveals when people are home, sleeping, or using specific appliances.

My objective was to build a distributed architecture where the forecasting model travels to the smart meters to train locally, so the raw data never leaves the house."

## 2. The Dataset

"For this experiment, I utilized the **UCI Individual Household Electric Power Consumption dataset**.

- **Size**: It contains over 2 million measurements continuously sampled over nearly 4 years from a single house in France.
- **Parameters**: It includes active power, reactive power, voltage, and 3 sub-metering values.
- **My Target**: My goal is to predict the `Global_active_power` (total household load) one hour into the future."

## 3. Data Processing & Feature Engineering

"I wrote a robust data pipeline using Pandas and Scikit-learn:

1. **Cleaning**: I interpolated missing values and resampled the raw minute-by-minute data into reliable hourly intervals.
2. **Feature Extraction (Crucial Step)**: Instead of just feeding the raw power curve, I engineered **Cyclical Multivariate Time Features**. I applied sine and cosine transformations to the hour of the day, day of the week, and month of the year. This gives the neural network mathematical context about daily routines and seasonal shifts without exploding the dimensionality.
3. **Sequencing**: I formulated it as a sliding window problem. The models look at the past 24 hours of multivariate data to predict the 25th hour's load."

## 4. Modeling Techniques

"I implemented two main architectures in **PyTorch**:

- **LSTM (Baseline)**: A standard Recurrent Neural Network.
- **TCN (Proposed)**: A Temporal Convolutional Network. It uses 1D causal convolutions with exponentially increasing dilations. TCNs are faster, avoid the vanishing gradient problem, and look back further into history than LSTMs.

To make the TCN extremely accurate, I deployed a **Residual Skip Connection**. Instead of forcing the AI to guess the total load from scratch, the network learns to predict only the _incremental change_ from the current hour to the next hour. This skyrocketed my accuracy."

## 5. Federated Learning Simulation

"To simulate the decentralized smart grid, I used the **Flower (`flwr`) framework**.

- Since my dataset came from one house, I had to synthesize artificial clients.
- I generated an **IID (Homogeneous)** dataset by randomly shuffling the sequences among 3 virtual clients.
- More importantly, I created a **Non-IID (Heterogeneous)** dataset by splitting the data chronologically (e.g., Client 1 gets Winter data, Client 2 gets Summer data). This perfectly mimics reality, where different houses have wildly different seasonal consumption patterns.
- I used the **FedAvg** algorithm on a central server to aggregate the local models."

## 6. The Results (The Proof)

"I designed an automated test runner to benchmark 5 specific configurations, evaluating them on Tracking Accuracy (Normalized RMSE). The results definitively prove my hypothesis:

1. **Centralized Baselines**: Both Centralized LSTM and Centralized TCN hovered around **84.6% accuracy**.
2. **Federated Parity (IID)**: The Federated TCN easily aggregated the remote data, achieving **91.24% accuracy**.
3. **The Non-IID Challenge**: Even under the harsh, realistic condition where the clients have totally different seasonal data distributions (Non-IID), my **Federated TCN** stayed incredibly robust, achieving **90.29% accuracy**.

**Conclusion**: My Federated TCN architecture not only preserves 100% of user data privacy but actually outperformed standard centralized LSTM models by over 5%, proving that decentralized convolutional forecasting is highly viable for modern smart grids."
