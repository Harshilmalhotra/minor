I am a B.Tech Computer Science student working on a minor research project with the goal of publishing a conference paper.

🎯 Project Focus

I want to build and evaluate a Federated Learning framework using Temporal Convolutional Networks (TCN) for short-term smart meter load forecasting under non-IID data conditions.

🔬 Core Research Question

Can a Federated TCN model accurately forecast short-term household electricity demand when data is heterogeneous (non-IID) across clients, and how does it compare to centralized training?

📊 Dataset

I am using the UCI Individual Household Electric Power Consumption dataset, which contains:

Date

Time

Global_active_power

Global_reactive_power

Voltage

Global_intensity

Sub_metering_1

Sub_metering_2

Sub_metering_3

Since this dataset contains data from a single household, I will simulate multiple federated clients by splitting the dataset to create heterogeneous (non-IID) distributions.

📌 Scope Restrictions (Do NOT Expand Beyond This)

Keep the project focused only on:

Short-term load forecasting (e.g., next 1-hour prediction)

Federated Learning (FedAvg) + flwr

Temporal Convolutional Networks (TCN)

Comparison with LSTM baseline

IID vs Non-IID client analysis

Do NOT expand into:

Renewable energy optimization

Reinforcement learning

IoT hardware implementation

Blockchain systems

Smart grid control/optimization

🏗 What I Need From You

Guide me step-by-step like a research supervisor to:

Part 1 — Project Implementation

Dataset preprocessing pipeline

Problem formulation (input-output structure)

Centralized LSTM implementation

Centralized TCN implementation

Federated Learning simulation using FedAvg

Creating IID and Non-IID client splits

Experimental design

Evaluation metrics (MAE, RMSE, MAPE)

Result analysis strategy

Part 2 — Research Paper Writing

How to write the Introduction properly

How to define the research gap clearly

Related Work section structure

Mathematical formulation of FedAvg and TCN

Experimental section writing

Result discussion & interpretation

Conclusion and future work

How to prepare figures and tables for publication

🎯 Expected Final Outcome

By the end, I want:

A working Federated TCN simulation

Comparison table including:

Centralized LSTM

Centralized TCN

Federated LSTM (IID)

Federated TCN (IID)

Federated TCN (Non-IID)

Quantitative evaluation

Clear experimental insights

A structured draft-ready research paper

Guide me clearly in phases, with milestones and realistic timelines.