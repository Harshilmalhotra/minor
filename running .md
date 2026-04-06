# Running the Federated Algorithm Comparison

## Step 1: Create a Virtual Environment (Mac)

```bash
cd ~/Desktop/minor
python3 -m venv venv
```

## Step 2: Activate the Virtual Environment

```bash
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the Federated Algorithm Comparison

```bash
python run_fed_algorithm_comparison.py
```

This will run all 10 federated learning algorithms (FedAvg, FedProx, FedNova, SCAFFOLD, FedOpt, FedAdam, FedYogi, FedAdagrad, FedDyn, Clustered FL) with the Cascade LSTM model on algorithmically generated client data, and save detailed results to:

- `results/fed_algorithm_comparison.csv` — Final comparison table
- `results/fed_algorithm_per_round.csv` — Per-round convergence data

## Step 5: Generate Comparison Plots (Optional)

```bash
python plot_fed_comparison.py
```

This generates bar chart comparisons and convergence curves in `results/plots/`.

## Note

Make sure `household_power_consumption.txt` is present in the project root directory before running the scripts